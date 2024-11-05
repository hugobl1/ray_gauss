from classes import optimizer


def define_optimizer_manager(config_opti,pointcloud,lr_list_pos,lr_list_rgb,lr_list_sh,lr_list_density,
                            lr_list_scale,lr_list_quaternion,
                            lr_list_sph_gauss_features,lr_list_bandwidth_sharpness,lr_list_lobe_axis):
  config_density,config_rgb,config_sh,config_pos,config_scale,config_quaternion,config_sph_gauss_features,config_bandwidth_sharpness,config_lobe_axis=config_opti.density,config_opti.rgb,config_opti.sh,config_opti.position,config_opti.scale,config_opti.quaternion,config_opti.sph_gauss_features,config_opti.bandwidth_sharpness,config_opti.lobe_axis
  if config_pos.scheduler:
    optim_manag_positions=optimizer.init_ExponentialLR(
    config_pos, pointcloud.positions, lr_decay_steps=(config_pos.decay_step), lr_delay_steps=config_pos.delay_step, lr_list=lr_list_pos,donotuse_steps=config_pos.donotuse_steps
    )#lr_decay_steps=20
  else:
    optim_manag_positions=optimizer.init_ConstantLR(
    config_pos, pointcloud.positions, lr_list=lr_list_pos,donotuse_steps=config_pos.donotuse_steps
    )
  if config_rgb.scheduler:
    optim_manag_rgb=optimizer.init_ExponentialLR(
      config_rgb, pointcloud.rgb, lr_decay_steps=(config_rgb.decay_step), lr_delay_steps=config_rgb.delay_step, lr_list=lr_list_rgb,donotuse_steps=config_rgb.donotuse_steps
    )
  else:
    optim_manag_rgb=optimizer.init_ConstantLR(
      config_rgb, pointcloud.rgb, lr_list=lr_list_rgb,donotuse_steps=config_rgb.donotuse_steps
    )
  if config_sh.scheduler:
    optim_manag_spherical_harmonics=optimizer.init_ExponentialLR(
      config_sh, pointcloud.spherical_harmonics, lr_decay_steps=(config_sh.decay_step), lr_delay_steps=config_sh.delay_step, lr_list=lr_list_sh,donotuse_steps=config_sh.donotuse_steps
    )
  else:
    optim_manag_spherical_harmonics=optimizer.init_ConstantLR(
      config_sh, pointcloud.spherical_harmonics, lr_list=lr_list_sh,donotuse_steps=config_sh.donotuse_steps
    )
  if config_density.scheduler:
    optim_manag_densities=optimizer.init_ExponentialLR(
      config_density, pointcloud.densities, lr_decay_steps=(config_density.decay_step), lr_delay_steps=config_density.delay_step, lr_list=lr_list_density,donotuse_steps=config_density.donotuse_steps
    )
  else:
    optim_manag_densities=optimizer.init_ConstantLR(
      config_density, pointcloud.densities, lr_list=lr_list_density,donotuse_steps=config_density.donotuse_steps
    )
  if config_scale.scheduler:
    optim_manag_scale=optimizer.init_ExponentialLR(
      config_scale,pointcloud.scales, lr_decay_steps=(config_scale.decay_step), lr_delay_steps=config_scale.delay_step, lr_list=lr_list_scale,donotuse_steps=config_scale.donotuse_steps
    )
  else:
    optim_manag_scale=optimizer.init_ConstantLR(
      config_scale,pointcloud.scales, lr_list=lr_list_scale,donotuse_steps=config_scale.donotuse_steps
    )
  if config_quaternion.scheduler:
    optim_manag_quaternion=optimizer.init_ExponentialLR(
      config_quaternion,pointcloud.quaternions, lr_decay_steps=(config_quaternion.decay_step), lr_delay_steps=config_quaternion.delay_step, lr_list=lr_list_quaternion,
        donotuse_steps=config_quaternion.donotuse_steps
    )
  else:
    optim_manag_quaternion=optimizer.init_ConstantLR(
      config_quaternion,pointcloud.quaternions, lr_list=lr_list_quaternion,donotuse_steps=config_quaternion.donotuse_steps
    )
  if config_sph_gauss_features.scheduler:
    optim_manag_sph_gauss_features=optimizer.init_ExponentialLR(
      config_sph_gauss_features,pointcloud.sph_gauss_features, lr_decay_steps=(config_sph_gauss_features.decay_step), lr_delay_steps=config_sph_gauss_features.delay_step, lr_list=lr_list_sph_gauss_features,donotuse_steps=config_sph_gauss_features.donotuse_steps
    )
  else:
    optim_manag_sph_gauss_features=optimizer.init_ConstantLR(
      config_sph_gauss_features,pointcloud.sph_gauss_features, lr_list=lr_list_sph_gauss_features,donotuse_steps=config_sph_gauss_features.donotuse_steps
    )
  if config_bandwidth_sharpness.scheduler:
    optim_manag_bandwidth_sharpness=optimizer.init_ExponentialLR(
      config_bandwidth_sharpness,pointcloud.bandwidth_sharpness, lr_decay_steps=(config_bandwidth_sharpness.decay_step), lr_delay_steps=config_bandwidth_sharpness.delay_step, lr_list=lr_list_bandwidth_sharpness,donotuse_steps=config_bandwidth_sharpness.donotuse_steps
    )
  else:
    optim_manag_bandwidth_sharpness=optimizer.init_ConstantLR(
      config_bandwidth_sharpness,pointcloud.bandwidth_sharpness, lr_list=lr_list_bandwidth_sharpness,donotuse_steps=config_bandwidth_sharpness.donotuse_steps
    )
  if config_lobe_axis.scheduler:
    optim_manag_lobe_axis=optimizer.init_ExponentialLR(
      config_lobe_axis,pointcloud.lobe_axis, lr_decay_steps=(config_lobe_axis.decay_step), lr_delay_steps=config_lobe_axis.delay_step, lr_list=lr_list_lobe_axis,donotuse_steps=config_lobe_axis.donotuse_steps
    )
  else:
    optim_manag_lobe_axis=optimizer.init_ConstantLR(
      config_lobe_axis,pointcloud.lobe_axis, lr_list=lr_list_lobe_axis,donotuse_steps=config_lobe_axis.donotuse_steps
    )
  return [optim_manag_positions,optim_manag_rgb,optim_manag_spherical_harmonics,optim_manag_densities,
          optim_manag_scale,optim_manag_quaternion,optim_manag_sph_gauss_features,optim_manag_bandwidth_sharpness,optim_manag_lobe_axis]
  
