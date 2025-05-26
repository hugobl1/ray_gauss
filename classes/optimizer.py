import torch
import matplotlib.pyplot as plt

class OptimizerManager:
    def __init__(self, optimizer, lr_scheduler=None, lr_decay_step=None, lr_delay_step=None,lr_list=[],donotuse_steps=0):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_decay_step = lr_decay_step
        self.lr_delay_step = lr_delay_step
        self.current_step=0
        self.lr_list = lr_list
        self.donotuse_steps=donotuse_steps
    def step(self):
        if self.current_step<self.donotuse_steps:
            self.current_step+=1
            return
        self.optimizer.step()
        if self.lr_scheduler is not None:
            if (self.current_step >= self.lr_delay_step)and(self.current_step<self.lr_delay_step+self.lr_decay_step):
                self.lr_scheduler.step()
        self.current_step+=1
    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)
    def save_lr(self):
        self.lr_list.append(self.optimizer.param_groups[0]['lr'])
    def save_lr_plot(self, path):
        plt.plot(self.lr_list)
        plt.savefig(path)
        plt.close()
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
    def print_state(self):
        print("Optimizer state: ",self.optimizer.state_dict())
        print("Optimizer state",self.optimizer.state)
        print("Optimizer param_groups: ",self.optimizer.param_groups)
        print("Optimizer state",self.optimizer.state)
        print("Optimizer lr: ",self.optimizer.param_groups[0]['lr'])
        # for group in self.optimizer.param_groups:
        #     a=self.optimizer.state.get(group['params'][0], None)
        #     print("a['square_avg'].shape",a['square_avg'].shape)
        
    # def _prune_optimizer(self, mask,new_tensor):
    #     for group in self.optimizer.param_groups:
    #         stored_state = self.optimizer.state.get(group['params'][0], None)
    #         if stored_state is not None:
    #             stored_state["square_avg"] = stored_state["square_avg"][mask]
    #             del self.optimizer.state[group['params'][0]]
    #             group["params"][0] = (new_tensor).requires_grad_(True)
    #             self.optimizer.state[group['params'][0]] = stored_state
    #         else:
    #             group["params"][0] = (group["params"][0][mask].requires_grad_(True))
        
    def _prune_optimizer(self, mask):
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                #stored_state["square_avg"] = stored_state["square_avg"][mask]
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = (group["params"][0][mask].requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
            else:
                group["params"][0] = (group["params"][0][mask].requires_grad_(True))
        return group["params"][0]
    
    def _cat_tensor(self, extension_tensor, dim):
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            extension_state=torch.zeros_like(extension_tensor)
            if stored_state is not None:
                #stored_state["square_avg"] = torch.cat((stored_state["square_avg"],extension_state),dim=dim)
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"],extension_state),dim=dim)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"],extension_state),dim=dim)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = (torch.cat((group["params"][0],extension_tensor),dim=dim).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
            else:
                group["params"][0] = (torch.cat((group["params"][0],extension_tensor),dim=dim).requires_grad_(True))
        return group["params"][0]

    
    def _get_state(self,name_state):
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                return stored_state[name_state]
            else:
                return None
    # def _replace(self, new_tensor):
    #     self.optimizer=torch.optim.Adam([new_tensor],lr=self.optimizer.param_groups[0]['lr'], eps=1e-15)
    #     if self.lr_scheduler is not None:
    #         self.lr_scheduler =  torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_scheduler.gamma)
    def _replace(self, new_tensor):
        group = self.optimizer.param_groups[0]
        stored_state = self.optimizer.state.get(group['params'][0], None)
        stored_state["exp_avg"] = torch.zeros_like(new_tensor)
        stored_state["exp_avg_sq"] = torch.zeros_like(new_tensor)
        del self.optimizer.state[group['params'][0]]
        group["params"][0] = new_tensor
        # group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))
        self.optimizer.state[group['params'][0]] = stored_state

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    def get_state_dict(self):
        return self.optimizer.state_dict()
    def load_state_dict(self, state_dict):
        return self.optimizer.load_state_dict(state_dict)
      


        
# Create an optimizer manager class that manages multiple optimizers and learning rate schedulers based on OptimizerManager
class OptimizerManagerDict:
    def __init__(self, dict_optimizer):
        self.optimizer_manager_dict=dict_optimizer
    def step(self):
        # for optimizer_manager in self.optimizer_manager_list:
        for optimizer_manager in self.optimizer_manager_dict.values():
            optimizer_manager.step()
    def zero_grad(self):
        # for optimizer_manager in self.optimizer_manager_list:
        for optimizer_manager in self.optimizer_manager_dict.values():
            optimizer_manager.zero_grad()
    def save_lr(self):
        # for optimizer_manager in self.optimizer_manager_list:
        for optimizer_manager in self.optimizer_manager_dict.values():
            optimizer_manager.save_lr()
    def save_lr_plot(self, path_list):
        # for optimizer_manager, path in zip(self.optimizer_manager_list, path_list):
            # optimizer_manager.save_lr_plot(path)
        for name, path in zip(self.optimizer_manager_dict.keys(), path_list):
            self.optimizer_manager_dict[name].save_lr_plot(path)
    def print_state(self):
        # for optimizer_manager in self.optimizer_manager_list:
        for optimizer_manager in self.optimizer_manager_dict.values():
            optimizer_manager.print_state()
    def _prune_optimizers(self, mask):
        dict={}
        for name, optimizer_manager in self.optimizer_manager_dict.items():
            pruned_tensor=optimizer_manager._prune_optimizer(mask)
            dict[name]=pruned_tensor
        return dict
    def _cat_tensors(self, extension_tensor_dict, dim):
        dict={}
        for name, extension_tensor in extension_tensor_dict.items():
            if name in self.optimizer_manager_dict.keys():
                cat_tensor=self.optimizer_manager_dict[name]._cat_tensor(extension_tensor,dim)
            else:
                exit("Warning: ",name," not in optimizer_manager_dict")
            dict[name]=cat_tensor
        return dict
    # def _prune_optimizers(self, mask,new_tensors_dict):
    #     # for optimizer_manager, new_tensor in zip(self.optimizer_manager_list, new_tensors):
    #     #     optimizer_manager._prune_optimizer(mask,new_tensor)
    #     # for name, new_tensor in zip(self.optimizer_manager_dict.keys(), new_tensors):
    #     #     self.optimizer_manager_dict[name]._prune_optimizer(mask,new_tensor)
    #     for name, new_tensor in new_tensors_dict.items():
    #         if name in self.optimizer_manager_dict.keys():
    #             self.optimizer_manager_dict[name]._prune_optimizer(mask,new_tensor)
    #         else:
    #             print("Warning: ",name," not in optimizer_manager_dict")          
    # def _cat_tensors(self, new_tensors_dict):
    #     for name,new_tensor in new_tensors_dict.items():
    #         if name in self.optimizer_manager_dict.keys():
    #             self.optimizer_manager_dict[name]._cat_tensors(new_tensor)
    #         else:
    #             print("Warning: ",name," not in optimizer_manager_dict")
    def _get_state(self,name_opti,name_state):
        state_value= self.optimizer_manager_dict[name_opti]._get_state(name_state)
        return state_value
    def _replace(self, dicts_replace):
        for name, tensor in dicts_replace.items():
            if name in self.optimizer_manager_dict.keys():
                self.optimizer_manager_dict[name]._replace(tensor)
            else:
                exit("Warning: ",name," not in optimizer_manager_dict")
    def get_state_dict(self):
        dict={}
        for name, optimizer_manager in self.optimizer_manager_dict.items():
            dict[name]=optimizer_manager.get_state_dict()
        return dict
    def load_state_dict(self, state_dict):
        for name, state in state_dict.items():
            if name in self.optimizer_manager_dict.keys():
                self.optimizer_manager_dict[name].load_state_dict(state)
            else:
                exit("Warning: ",name," not in optimizer_manager_dict")
         
def init_ConstantLR(config, optimize_tensor, lr_list, donotuse_steps=0):
    optimizer=torch.optim.Adam([optimize_tensor],lr=config.init_lr, eps=1e-15)
    return OptimizerManager(optimizer,lr_scheduler=None,lr_list=lr_list,donotuse_steps=donotuse_steps)
   
def init_ExponentialLR(config, optimize_tensor, lr_decay_steps, lr_delay_steps, lr_list, donotuse_steps=0):
    #check if config.gamma is present
    if not hasattr(config, 'gamma'):
        init_lr=config.init_lr
        final_lr=config.final_lr
        gamma=(final_lr/init_lr)**(1/lr_decay_steps)
    else:
        init_lr=config.init_lr
        gamma=config.gamma
    optimizer=torch.optim.Adam([optimize_tensor],lr=init_lr, eps=1e-15)
    scheduler =  torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    return OptimizerManager(optimizer,scheduler,lr_decay_steps,lr_delay_steps,lr_list,donotuse_steps)

def init_StepLR(config, optimize_tensor, lr_decay_steps, lr_delay_steps, step_size, lr_list):
    #check if config.gamma is present
    if not hasattr(config, 'gamma'):
        init_lr=config.init_lr
        final_lr=config.final_lr
        gamma=(final_lr/init_lr)**(1/lr_decay_steps)
    else:
        init_lr=config.init_lr
        gamma=config.gamma
    optimizer=torch.optim.Adam([optimize_tensor],lr=init_lr, eps=1e-15)
    scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return OptimizerManager(optimizer,scheduler,lr_decay_steps,lr_delay_steps,lr_list)


