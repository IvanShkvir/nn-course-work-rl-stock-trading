# Neural Networks. Course work
## Simple RL models for crypto stock trading

In this work I did an investigation about how to apply neural networks for automatic trading on the crypto stock. I have created the custom OpenAI gym like environment which can be used to simulate the stocks trading. For the base of the environment was used `gym_anytrading` library and its absctract environement `TradingEnv`. The data for environment was collected using `binance` library. As for the models I have used `stable_baselines3` library which provide various RL (reinforcment learning) models such as DQN, PPO, A2C and others. As well I needed the LSTM version PPO model so I used unstable contributed version of stable_baselines3 called `sb3_contrib`. For visualising the outputs I have chosen library `quantstats` which provide a simple way to generate extensive html reports about the profit history. 

As a result were created 4 models: random (with random actions), DQN, PPO and LSTM PPO. All of them were visualized and evaluated. Trained models are stored in `models` folder, results are stored in `reports` folder.

If you want to repeat my experience and results you can clone the repository and simply execute the IPython notebook `main.ipynb`. To change datasets you have to pass your Binance API credentials to utils file (variables `api_key` and `api_secret`), delete existing datasets and call method `get_ready_dataframe` with other arguments. By default the notebook uses already trained models from `models` folder, if you want to retrain them you can simply uncomment corresponding lines in the notebook.

That's it. Hope you enjoy my small project :)
