using System;
using System.Linq;

namespace Ctrnn.Simulation
{
	sealed class NeuralNet
	{
		public const double TimeConstantLowRange = 1.0;
		public const double TimeConstantHighRange = 20.0;
		public const double GainLowRange = 0.0;
		public const double GainHighRange = 0.5;
		public const double BiasLowRange = -10.0;
		public const double BiasHighRange = 0.0;
		public const double InputScalingLowRange = -10.0;
		public const double InputScalingHighRange = 10.0;
		public const double WeightLowRange = -5.0;
		public const double WeightHighRange = 5.0;

		public NeuralNet(
			int n,
			double[] timeConstant, double[] gain, double[] bias, double[] inputScaling, double[][] weight
			)
		{
			if (n <= 0)
			{
				throw new ArgumentOutOfRangeException(nameof(n));
			}

			if (timeConstant == null)
			{
				throw new ArgumentNullException(nameof(timeConstant));
			}
			if (timeConstant.Length != n)
			{
				throw new ArgumentOutOfRangeException(nameof(timeConstant));
			}
			ValidateRange(timeConstant, TimeConstantLowRange, TimeConstantHighRange, nameof(timeConstant));

			if (gain == null)
			{
				throw new ArgumentNullException(nameof(gain));
			}
			if (gain.Length != n)
			{
				throw new ArgumentOutOfRangeException(nameof(gain));
			}
			ValidateRange(gain, GainLowRange, GainHighRange, nameof(gain));

			if (bias == null)
			{
				throw new ArgumentNullException(nameof(bias));
			}
			if (bias.Length != n)
			{
				throw new ArgumentOutOfRangeException(nameof(bias));
			}
			ValidateRange(bias, BiasLowRange, BiasHighRange, nameof(bias));

			if (inputScaling == null)
			{
				throw new ArgumentNullException(nameof(inputScaling));
			}
			if (inputScaling.Length != n)
			{
				throw new ArgumentOutOfRangeException(nameof(inputScaling));
			}
			ValidateRange(inputScaling, InputScalingLowRange, InputScalingHighRange, nameof(inputScaling));

			if (weight == null)
			{
				throw new ArgumentNullException(nameof(weight));
			}
			if (weight.Length != n)
			{
				throw new ArgumentOutOfRangeException(nameof(weight));
			}
			for (var i = 0; i < n; ++i)
			{
				if (weight[i] == null)
				{
					throw new ArgumentNullException($"{nameof(weight)}[{i}]");
				}
				if (weight[i].Length != n)
				{
					throw new ArgumentOutOfRangeException($"{nameof(weight)}[{i}]");
				}
				ValidateRange(weight[i], WeightLowRange, WeightHighRange, $"{nameof(weight)}[{i}]");
			}

			_n = n;
			_state = new double[_n];
			_input = new double[_n];
			_timeConstant = (double[])timeConstant.Clone();
			_gain = (double[])gain.Clone();
			_bias = (double[])bias.Clone();
			_inputScaling = (double[])inputScaling.Clone();

			_weight = new double[_n][];

			for (var i = 0; i < n; ++i)
			{
				_weight[i] = (double[])weight[i].Clone();
			}
		}

		/// <summary>
		/// Get the state of the nth neuron.
		/// </summary>
		public double GetState(int n)
		{
			if (n < 0 || n > _n)
			{
				throw new ArgumentOutOfRangeException(nameof(n));
			}

			return _state[n];
		}

		/// <summary>
		/// Get the assigned input value of the nth neuron.
		/// </summary>
		public double GetInput(int n)
		{
			if (n < 0 || n > _n)
			{
				throw new ArgumentOutOfRangeException(nameof(n));
			}

			return _state[n];
		}

		/// <summary>
		/// Set the input value given to the nth neuron.
		/// </summary>
		public void SetInput(int n, double value)
		{
			if (n < 0 || n > _n)
			{
				throw new ArgumentOutOfRangeException(nameof(n));
			}

			_input[n] = value;
		}

		/// <summary>
		/// Update the state of all neurons over a single time-slice.
		/// </summary>
		public void Update()
		{
			double[] yd = new double[_n];

			for (var i = 0; i < _n; ++i)
			{
				var ydi = -_state[i];

				for (var j = 0; j < _n; ++j)
				{
					var zj = Sigma(_gain[j] * (_state[j] + _bias[j]));

					ydi += zj * _weight[i][j];
				}

				ydi += _inputScaling[i] * _input[i];

				yd[i] = ydi;
			}

			for (var i = 0; i < _n; ++i)
			{
				_state[i] += yd[i] / _timeConstant[i];
			}
		}

		private static void ValidateRange(double[] values, double low, double high, string name)
		{
			if (values.Any(v => v < low || v > high))
			{
				throw new ArgumentOutOfRangeException($"Values for '{name}' must be between {low} and {high}.");
			}
		}

		/// <summary>
		/// Implements the standard sigmoidal logistic or activation function.
		/// </summary>
		private static double Sigma(double x)
		{
			return 1.0 / (1.0 + Math.Pow(Math.E, x));
		}

		private readonly int _n;
		private readonly double[] _state;
		private readonly double[] _input;
		private readonly double[] _timeConstant;
		private readonly double[] _gain;
		private readonly double[] _bias;
		private readonly double[] _inputScaling;
		private readonly double[][] _weight;
	}
}
