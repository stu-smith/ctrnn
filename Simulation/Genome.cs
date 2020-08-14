using System;

namespace Ctrnn.Simulation
{
	sealed class Genome
	{
		private Genome(int n)
		{
			if (n <= 0)
			{
				throw new ArgumentOutOfRangeException(nameof(n));
			}

			var nv
				= 4 * n // timeConstant, gain, bias, inputScaling : total four lots
				+ n * n // full-interconnect weights
				;

			_n = n;
			_values = new double[nv];
		}

		public Genome FromRandom(int n, Random rnd)
		{
			var g = new Genome(n);

			for (var i = 0; i < g._values.Length; ++i)
			{
				g._values[i] = rnd.NextDouble();
			}

			return g;
		}

		public NeuralNet Express()
		{
			var timeConstant = ScaleRange(_n * 0, NeuralNet.TimeConstantLowRange, NeuralNet.TimeConstantHighRange);
			var gain = ScaleRange(_n * 1, NeuralNet.GainLowRange, NeuralNet.GainHighRange);
			var bias = ScaleRange(_n * 2, NeuralNet.BiasLowRange, NeuralNet.BiasHighRange);
			var inputScaling = ScaleRange(_n * 4, NeuralNet.InputScalingLowRange, NeuralNet.InputScalingHighRange);

			var weights = new double[_n][];

			for (var i = 0; i < _n; ++i)
			{
				weights[i] = ScaleRange(_n * (5 + i), NeuralNet.WeightLowRange, NeuralNet.WeightHighRange);
			}

			return new NeuralNet(_n, timeConstant, gain, bias, inputScaling, weights);
		}

		public Genome Mutate(Random rnd, double stdDev)
		{
			var g = new Genome(_n);

			for(var i = 0; i < _values.Length; ++i)
			{
				g._values[i] = Math.Clamp(_values[i] + BoxMuller(rnd, stdDev), 0.0, 1.0);
			}

			return g;
		}

		private double[] ScaleRange(int start, double lowRange, double highRange)
		{
			var values = new double[_n];

			for(var i = 0; i < _n; ++i)
			{
				values[i] = _values[start + i] * (highRange - lowRange) + lowRange;
			}

			return values;
		}

		private static double BoxMuller(Random rnd, double stdDev)
		{
			var u1 = rnd.NextDouble();
			var u2 = rnd.NextDouble();
			var rndStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
			var rndNormal = stdDev * rndStdNormal;

			return rndNormal;
		}

		private int _n;
		private readonly double[] _values;
	}
}
