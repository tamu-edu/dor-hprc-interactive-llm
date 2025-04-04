# User Documentation
## Installing on HPRC Launch
```
git clone git@github.com:tamu-edu/dor-hprc-interactive-llm.git
cd dor-hprc-interactive-llm
source modules.sh
python3 -m venv venv
pip install -r requirements.txt
pip install -e jupyter-ai/packages/HPRC-llama-8B/
```
## Using in Jupyterlab on HPRC Launch
Create a new jupyter lab session, make sure you select the Python/3.10.8 module.  
Specify the path to the venv you created, should look something like:  
```
$SCRATCH/dor-hprc-interactive-llm/venv/bin/activate
```
All other parameters are arbitrary.  
Once in your jupyterlab session, click on the conversation icon on the left side of the screen:  
![]()
Click "start here" if that appears. 
Select one of the HPRC providers for the completion model parameter, and leave other parameters unchanged.  
Click "save changes" and navigate back with the back arrow.  
You should now see a terminal where you can enter slash commands to interact with the model.  
