# Tutorial: Python Breakpoint Debugging on HPC

If you are tired of debugging by print statements, please read.

# 1. Configuring SSH on Your Local Machine

One of the major road blocks is that when you `salloc` on HPC, you will need an additional ssh from `esplhpccompbio-lv01` to log into the compute node. Under normal circumstances, this will prevent you from connecting to the compute note and running python debug from VS code or Cursor. Let’s fix this.

First, assuming you are on a mac, copy the following to a file located at `$HOME/.ssh/config`. If you don’t have this file, simply create it. Substitute `<your-username>` with your Cedars usersname.

```ini
Host lv01
    HostName esplhpccompbio-lv01
    User <your-username>
Host cp*
    Hostname esplhpc-%h
    User <your-username>
    ProxyJump lv01
```

  
  

Then, after you run `salloc` on HPC and have been assigned a compute node, for example, `esplhpc-cp076`. You can simply login to the assigned compute node by

```bash 
ssh cp076
```

> If you decide logging into HPC systems with password is too much of a hassel, the checkout the [password less ssh](https://www.redhat.com/en/blog/passwordless-ssh "https://www.redhat.com/en/blog/passwordless-ssh").

----------

# 2. Configure VS Code to Run Python Debug

1. you should have [enabled remote development with ssh](https://code.visualstudio.com/docs/remote/ssh-tutorial "https://code.visualstudio.com/docs/remote/ssh-tutorial") on your VS code. Connect to your allocated compute note by simply typing `cp<node-number>` for the SSH host. Then open the appropriate folder for your project.

2. Install the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [Python Debugger](https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy) extensions on your remote machine.

3. Create a folder named `.vscode` in your project folder. Add the following two JSON files in the `.vscode` folder.

   `launch.json`:

   ```json
   {
     "version": "0.2.0",
     "configurations": [
       {
         "name"   : "Python Debugger: Current File",
         "type"   : "debugpy",
         "request": "launch",
         "program": "${file}",
         "console": "integratedTerminal"
       },
       {
         "name": "PromptMRI Inference",
         "type": "debugpy",
         "request": "launch",
         "postDebugTask": "closeTerminal",
         "program": "main.py",
         "args": [
           "predict",
           "--config",
           "configs/inference/pmr-plus/cmr25-task2-val-saveitr.yaml"
         ],
         "env": {"CUDA_VISIBLE_DEVICES": "1"},
         "console": "integratedTerminal"
       }
     ]
   }
   ```

   `tasks.json`:

   ```json
   {
     "version": "2.0.0",
     "tasks": [
       {
         "label": "closeTerminal",
         "type": "process",
         "command": "${command:workbench.action.terminal.kill}",
         "problemMatcher": []
       }
     ]
   }
   ```

   In this example, `launch.json` provides two python debug configurations `Python Debugger: Current File` and `PromptMRI Inference`. The `PromptMRI Inference` will launch the debugger as if you called python with `python main.py --config configs/inference/pmr-plus/cmr25-task2-val-saveitr.yaml`. Customize `args` to your needs. Create additional configurations should you choose to.

   For your convenience, `PromptMRI Inference` also has a `postDebugTask`, which will close the debug console per `tasks.json`.

   Read more: [[VS Code Debugging](https://code.visualstudio.com/docs/python/debugging "https://code.visualstudio.com/docs/python/debugging")] [[VS Code Tasks](https://code.visualstudio.com/docs/debugtest/tasks "https://code.visualstudio.com/docs/debugtest/tasks")]
  

----------

# 3. Launching Python Debug

1.  After you have opened your project on the remote compute node, you should see a `Run and Debug` module on the top left corner go into it.
    


<p align="center" width="100%">
    <img width="5%" src="https://junzhou.chen.engineer/Resources/debug_symbol.png">
</p>


2.  You should be able to see a list of debug configurations, choose the one you would like to run:  
      
    

<p align="center" width="100%">
    <img width="30%" src="https://junzhou.chen.engineer/Resources/debug_configs_selection.png">
</p>

3.  [⚠️ Important] On the bottom right corner, select the appropriate conda environment.
    
<p align="center" width="100%">
    <img width="30%" src="https://junzhou.chen.engineer/Resources/conda_env_selection.png">
</p>

4.  Set breakpoints by clicking left to the line number in your python file.
    

<p align="center" width="100%">
    <img width="100%" src="https://junzhou.chen.engineer/Resources/breakpoint.png">
</p>
5.  Hit ▶️ and run. Enjoy
    
<p align="center" width="100%">
    <img width="100%" src="https://junzhou.chen.engineer/Resources/debug_UI.png">
</p>