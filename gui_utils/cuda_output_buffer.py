import sys, os, enum
from packaging import version

import numpy as np
import cupy as cp

import OpenGL.GL as gl

from .vecmath import vtype_to_dtype

try:
    import cuda as _cuda
    from cuda import cudart
    has_cudart = True
    has_gl_interop = version.parse(_cuda.__version__) >= version.parse("11.6.0")
except ImportError:
    cudart = None
    has_cudart = False
    has_gl_interop = False

_cuda_opengl_interop_msg = (
    "Cuda Python low level bindings v11.6.0 or later are required to enable "
   f"Cuda/OpenGL interoperability.{os.linesep}You can install the missing package with:"
   f"{os.linesep}  {sys.executable} -m pip install --upgrade --user cuda-python"
)

if has_cudart:
    def format_cudart_err(err):
        return (
            f"{cudart.cudaGetErrorName(err)[1].decode('utf-8')}({int(err)}): "
            f"{cudart.cudaGetErrorString(err)[1].decode('utf-8')}"
        )


    def check_cudart_err(args):
        if isinstance(args, tuple):
            assert len(args) >= 1
            err = args[0]
            if len(args) == 1:
                ret = None
            elif len(args) == 2:
                ret = args[1]
            else:
                ret = args[1:]
        else:
            ret = None

        assert isinstance(err, cudart.cudaError_t), type(err)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(format_cudart_err(err))

        return ret


class BufferImageFormat(enum.Enum):
    UCHAR4=0
    FLOAT3=1
    FLOAT4=2

    @property
    def dtype(self):
        if self is BufferImageFormat.UCHAR4:
            return vtype_to_dtype('uchar4')
        elif self is BufferImageFormat.FLOAT3:
            return vtype_to_dtype('float3')
        elif self is BufferImageFormat.FLOAT4:
            return vtype_to_dtype('float4')
        else:
            raise NotImplementedError(self)

    @property
    def itemsize(self):
        return self.dtype.itemsize


class CudaOutputBufferType(enum.Enum):
    CUDA_DEVICE = 0,  # not preferred, typically slower than ZERO_COPY
    GL_INTEROP  = 1,  # single device only, preferred for single device
    ZERO_COPY   = 2,  # general case, preferred for multi-gpu if not fully nvlink connected
    CUDA_P2P    = 3,  # fully connected only, preferred for fully nvlink connected

    @classmethod
    def enable_gl_interop(cls, fallback=True):
        if has_gl_interop:
            return cls.GL_INTEROP
        elif fallback:
            msg = _cuda_opengl_interop_msg + f"{os.linesep}Falling back to slower CUDA_DEVICE output buffer."
            print(msg)
            return cls.CUDA_DEVICE
        else:
            raise RuntimeError(_cuda_opengl_interop_msg)


class CudaOutputBuffer:
    __slots__ = ['_pixel_format', '_buffer_type', '_width', '_height',
            '_device', '_device_idx', '_device', '_stream',
            '_host_buffer', '_device_buffer', '_cuda_gfx_ressource', '_pbo']

    def __init__(self, buffer_type, pixel_format, width, height, device_idx=0):
        for attr in self.__slots__:
            setattr(self, attr, None)

        self.device_idx = device_idx
        self.pixel_format = pixel_format
        self.buffer_type = buffer_type
        self.resize(width, height)
        self.stream = None
        
        if buffer_type is CudaOutputBufferType.GL_INTEROP:
            if not has_gl_interop:
                raise RuntimeError(_cuda_opengl_interop_msg)
            device_count, device_ids = check_cudart_err( cudart.cudaGLGetDevices(1, cudart.cudaGLDeviceList.cudaGLDeviceListAll) )
            if device_count <= 0:
                raise RuntimeError("No OpenGL device found, cannot enable GL_INTEROP.")
            elif device_ids[0] != device_idx:
                raise RuntimeError(f"OpenGL device id {device_ids[0]} does not match requested "
                                   f"device index {device_idx} for Cuda/OpenGL interop.")

        self._reallocate_buffers()

    def resize(self, width, height):
        self.width = width
        self.height = height

    def get_host_buffer(self):
        if self.buffer_type is CudaOutputBufferType.CUDA_DEVICE:
            self.copy_device_to_host()
            return self._host_buffer
        else:
            msg = f'Buffer type {self.buffer_type} has not been implemented yet.'
            raise NotImplementedError(msg)

    def map(self):
        self._make_current()
        if (self._host_buffer is None) or (self._device_buffer is None):
            self._reallocate_buffers()
        if self.buffer_type is CudaOutputBufferType.CUDA_DEVICE:
            return self._device_buffer.data.ptr
        elif self.buffer_type is CudaOutputBufferType.GL_INTEROP:
            check_cudart_err(
                cudart.cudaGraphicsMapResources(1, self._cuda_gfx_ressource, self._stream.ptr)
            )
            ptr, size = check_cudart_err(
                cudart.cudaGraphicsResourceGetMappedPointer(self._cuda_gfx_ressource)
            )
            return ptr
        else:
            msg = f'Buffer type {self.buffer_type} has not been implemented yet.'
            raise NotImplementedError(msg)

    def unmap(self):
        self._make_current()
        buffer_type = self.buffer_type
        if buffer_type is CudaOutputBufferType.CUDA_DEVICE:
            self._stream.synchronize()
        elif buffer_type is CudaOutputBufferType.GL_INTEROP:
            check_cudart_err(
                cudart.cudaGraphicsUnmapResources(1, self._cuda_gfx_ressource, self._stream.ptr)
            )
        else:
            msg = f'Buffer type {buffer_type} has not been implemented yet.'
            raise NotImplementedError(msg)

    def get_pbo(self):
        buffer_type = self.buffer_type

        self._make_current()

        if buffer_type is CudaOutputBufferType.CUDA_DEVICE:
            if self._pbo is None:
                self._pbo = gl.glGenBuffers(1)
            self.copy_device_to_host()
            self.copy_host_to_pbo()
        elif buffer_type is CudaOutputBufferType.GL_INTEROP:
            assert self._pbo is not None
        else:
            msg = f'Buffer type {buffer_type} has not been implemented yet.'
            raise NotImplementedError(msg)

        return self._pbo

    def delete_pbo(self):
        if self._pbo is None:
            return
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glDeleteBuffers(1, self._pbo)
        self._pbo = None

    def copy_device_to_host(self):
        cp.cuda.runtime.memcpy(self._host_buffer.__array_interface__['data'][0],
                self._device_buffer.data.ptr, self._host_buffer.nbytes, cp.cuda.runtime.memcpyDeviceToHost)

    def copy_host_to_pbo(self):
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._pbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self._host_buffer, gl.GL_STREAM_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def _make_current(self):
        self._device.use()

    def _reallocate_buffers(self):
        buffer_type = self.buffer_type

        dtype = self.pixel_format
        shape = (self.height, self.width)
        
        self._host_buffer = np.empty(shape=shape, dtype=dtype)

        if buffer_type is CudaOutputBufferType.CUDA_DEVICE:
            self._device_buffer = cp.empty(shape=shape, dtype=dtype)
            if self._pbo is not None:
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._pbo)
                gl.glBufferData(gl.GL_ARRAY_BUFFER, self._host_buffer, gl.GL_STREAM_DRAW)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        elif buffer_type is CudaOutputBufferType.GL_INTEROP:
            self._pbo = gl.glGenBuffers(1) if self._pbo is None else self._pbo
            
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._pbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, self.width*self.height*dtype.itemsize, None, gl.GL_STREAM_DRAW)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        
            self.cuda_gfx_ressource = check_cudart_err(
                cudart.cudaGraphicsGLRegisterBuffer(self._pbo,
                    cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard)
            )
        else:
            msg = f'Buffer type {buffer_type} has not been implemented yet.'
            raise NotImplementedError(msg)

    def _get_pixel_format(self):
        return self._pixel_format
    def _set_pixel_format(self, value):
        if value is None:
            value = BufferImageFormat.UCHAR4
        if isinstance(value, BufferImageFormat):
            value = value.dtype
        elif isinstance(value, str):
            value = vtype_to_dtype(value)
        assert isinstance(value, np.dtype) or issubclass(value, np.generic), value
        if value != getattr(self, '_pixel_format', None):
            self._pixel_format = value
            self._host_buffer = None
            self._device_buffer = None
    pixel_format = property(_get_pixel_format, _set_pixel_format)

    def _get_buffer_type(self):
        return self._buffer_type
    def _set_buffer_type(self, value):
        if value is None:
            value = CudaOutputBufferType.CUDA_DEVICE
        assert isinstance(value, CudaOutputBufferType), type(value)
        if value != getattr(self, '_buffer_type', None):
            self._buffer_type = value
            self._host_buffer = None
            self._device_buffer = None
    buffer_type = property(_get_buffer_type, _set_buffer_type)

    def _get_width(self):
        return self._width
    def _set_width(self, value):
        if value is None:
            value = 1
        assert value >= 1, value
        try:
            value = np.int32(np.asscalar(value))
        except AttributeError:
            value = np.int32(value)
        if value != getattr(self, '_width', None):
            self._width = value
            self._host_buffer = None
            self._device_buffer = None
    width = property(_get_width, _set_width)

    def _get_height(self):
        return self._height
    def _set_height(self, value):
        if value is None:
            value = 1
        assert value >= 1, value
        try:
            value = np.int32(np.asscalar(value))
        except AttributeError:
            value = np.int32(value)
        if value != getattr(self, '_height', None):
            self._height = value
            self._host_buffer = None
            self._device_buffer = None
    height = property(_get_height, _set_height)

    def _get_device_idx(self):
        return self._device
    def _set_device_idx(self, value):
        if value is None:
            value = 0
        assert value >= 0, value
        value = int(value)
        if value != getattr(self, '_device_idx', None):
            self._device_idx = value
            self._device = cp.cuda.Device(value)
            self._host_buffer = None
            self._device_buffer = None
    device_idx = property(_get_device_idx, _set_device_idx)

    def _get_stream(self):
        return self._stream
    def _set_stream(self, value):
        if value is None:
            value = cp.cuda.Stream.null
        assert isinstance(value, cp.cuda.Stream), type(value)
        self._stream = value
    stream = property(_get_stream, _set_stream)
    
    def _get_cuda_gfx_ressource(self):
        assert self._cuda_gfx_ressource is not None
        return self._cuda_gfx_ressource
    def _set_cuda_gfx_ressource(self, value):
        if (self._cuda_gfx_ressource is not None) and (self._cuda_gfx_ressource != value):
            check_cudart_err(
                cudart.cudaGraphicsUnregisterResource(self._cuda_gfx_ressource)
            )
        self._cuda_gfx_ressource = value

    cuda_gfx_ressource = property(_get_cuda_gfx_ressource, _set_cuda_gfx_ressource)
