The idea is to show that an adaptive method would perform well with one set of hyperparameters, compared to linear/exponential decay.

Results to find:

For each environment: HalfCheetah, Swimmer, Hopper, Reacher

	KL:
		Linear:
			decrease KL by 0.000001, 0.00001, 0.0001
		Exponential:
			decrease KL by 0.9, 0.99, 0.999
		Adaptive:
			adapt KL by 0.000001, 0.00001, 0.0001

	Stepcount:
		Linear:
			increase stepcount by 20, 100, 500, 1000
		Exponential:
			increase linear by 1.001, 1.01, 1.1
		Adaptive:
			adapt stepcount by 20, 100, 500, 1000

First only change the KL, then only change the stepcount.

We will make one final comparison between [only changing KL, only changing stepcount, and changing both] to show the adaptive method works when changing both hyperparameters