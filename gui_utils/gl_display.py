import ctypes
import numpy as np

import OpenGL.GL as gl
import OpenGL.GL.shaders

from .cuda_output_buffer import BufferImageFormat

class GLDisplay:
    vert_source = \
"""
#version 330 core

layout(location = 0) in vec3 position;
out vec2 UV;

void main()
{
    gl_Position =  vec4(position, 1);
    UV = (vec2(position.x, position.y) + vec2(1,1)) / 2.0;
}
"""

    frag_source = \
"""
#version 330 core

in vec2 UV;
layout(location=0) out vec4 color;

uniform sampler2D render_tex;

void main()
{
    color = texture(render_tex, UV).xyzw;
}
"""

    quad_vertex_buffer_data = np.asarray([
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [-1.0,  1.0, 0.0],
        [-1.0,  1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [ 1.0,  1.0, 0.0],
        ], dtype=np.float32)

    __slots__ = ['_image_format', '_render_tex', '_program', '_render_tex_uniforloc',
            '_quad_vertex_buffer', '_image_format']

    def __init__(self, image_format):
        assert isinstance(image_format, BufferImageFormat)

        vertex_array = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vertex_array)

        program = self.create_gl_program()
        render_tex_uniforloc = gl.glGetUniformLocation(program, "render_tex")

        render_tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, render_tex)

        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        quad_vertex_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, quad_vertex_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER,
            self.quad_vertex_buffer_data,
            gl.GL_STATIC_DRAW)

        self._image_format = image_format
        self._program = program
        self._render_tex = render_tex
        self._render_tex_uniforloc = render_tex_uniforloc
        self._quad_vertex_buffer = quad_vertex_buffer

    @classmethod
    def create_gl_program(cls):
        return gl.shaders.compileProgram(
            gl.shaders.compileShader(cls.vert_source, gl.GL_VERTEX_SHADER),
            gl.shaders.compileShader(cls.frag_source, gl.GL_FRAGMENT_SHADER),
        )

    def display(self, screen_res_x, screen_res_y, framebuf_res_x, framebuf_res_y, pbo):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, framebuf_res_x, framebuf_res_y)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glUseProgram(self._program)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._render_tex)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo)

        itemsize = self._image_format.itemsize
        if (itemsize % 8 == 0):
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 8)
        elif (itemsize % 4 == 0):
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
        elif (itemsize % 2 == 0):
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 2)
        else:
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

        convert_to_srgb = True

        image_format = self._image_format
        if(image_format == BufferImageFormat.UCHAR4):
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, screen_res_x, screen_res_y,
                            0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
            convert_to_srgb = False
        elif image_format is BufferImageFormat.FLOAT3:
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB32F, screen_res_x, screen_res_y,
                            0, gl.GL_RGB, gl.GL_FLOAT, None)
        elif image_format is BufferImageFormat.FLOAT4:
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, screen_res_x, screen_res_y,
                            0, gl.GL_RGBA, gl.GL_FLOAT, None)
        else:
            raise NotImplementedError(f"Unknown image format {image_format}.")

        if convert_to_srgb:
            gl.glEnable(gl.GL_FRAMEBUFFER_SRGB)
        else:
            gl.glDisable(gl.GL_FRAMEBUFFER_SRGB)

        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        gl.glUniform1i(self._render_tex_uniforloc, 0)

        # 1st attribute buffer : vertices
        gl.glEnableVertexAttribArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._quad_vertex_buffer)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, ctypes.c_void_p(0))
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glDisableVertexAttribArray(0)

        gl.glDisable(gl.GL_FRAMEBUFFER_SRGB)
