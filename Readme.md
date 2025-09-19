
This is the code for the paper
"Optimizing Quantum Circuits via ZX Diagrams using Reinforcement Learning and Graph Neural Networks"

The implementation here covers the setting using the features described in Appendix C.
This variant scales better than the graph based method and can be easily used for peephole optimization.

You can train the model using 
``python -u main_tree.py``

You may not be able to run the model without setting the appropriate settings:
The settings can be changed by using a custom hydra config, or by overwriting the values directly on the command line

``python -u main_tree.py +algorithm=PPO exp_name="20MIO_32_envs_5qubit_128treesize" +model=GATActionModel +env=more_complex_more_rules_ranges env.num_envs=32   model.model_type="ActionAtt" model.n_message_passing=4 algorithm.total_timesteps=20_000_000 algorithm.num_steps=129 device="cpu" max_tree_size=128  multi_range=4 algorithm.learning_rate=3e-3 env.n_qubits=5``

We use ray for parallelization, so you would need to set up a ray cluster for large scale training (see https://docs.ray.io/en/latest/ray-overview/getting-started.html).
We also have test scripts to compare compilers in bench_compilers.py Notice that this relies on Staudacher et al's pyzx optimizer for benchmarking which we do not distribute here.
You can either include that code or comment out the relevant lines in bench_compilers.py
You will also need the the pyzx environment found at https://github.com/MattAlexMiracle/pyzx_environment/tree/main to run the optimization.