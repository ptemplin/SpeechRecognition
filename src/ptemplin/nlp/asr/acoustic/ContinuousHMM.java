package ptemplin.nlp.asr.acoustic;

import java.math.BigDecimal;

import ptemplin.nlp.asr.util.BigProb;
import ptemplin.nlp.asr.util.Gaussian;

public class ContinuousHMM extends HiddenMarkovModel {
		
	private static final double[] DEFAULT_MEANS = 		{400, 500, 450, 500, 200, 50, -50, 75, 100, 40, 20, 10, 5};
	private static final double[] DEFAULT_VARIANCES = 	{200, 200, 200, 100, 100, 20, 20, 30, 50, 30, 50, 10, 10};

	private final double[][][] outputMeans;
	private final double[][][][] outputVariances;

	public ContinuousHMM() {
		super();
		outputMeans = new double[numStates][numStates][observationSize];
		outputVariances = new double[numStates][numStates][observationSize][observationSize];
		flatInitialize();
	}
	
	public double[][][] getOutputMeans() { return outputMeans; }
	public double[][][][] getOutputVariances() { return outputVariances; }

	private void flatInitialize() {
		// initialize transition probabilities equally
		for (int i = 0; i < numStates-1; i++) {
			for (int j = i; j <= i+1; j++) {
				transitionProbs[i][j] = 1.0/2;
			}
		}
		transitionProbs[numStates-1][numStates-1] = 1;
		// initialize output means equally
		for (int i = 0; i < numStates; i++) {
			for (int j = i; j<=i+1 && j < numStates; j++) {
				for (int k = 0; k < observationSize; k++) {
					outputMeans[i][j][k] = DEFAULT_MEANS[k];
				}
			}
		}
		// initialize variances equally
		for (int i = 0; i < numStates; i++) {
			for (int j = i; j<=i+1 && j < numStates; j++) {
				for (int k = 0; k < observationSize; k++) {
					outputVariances[i][j][k][k] = DEFAULT_VARIANCES[k];
				}
			}
		}
	}

	@Override
	public void updateOutputParameters(BigProb[][][] arcProbs, double[][] observationSeq) {
		final int totalTime = arcProbs[0][0].length;
		for (int i = 0; i < numStates; i++) {
			for (int j = i; j <= i+1 && j < numStates; j++) {
				// first calculate the denominator
				BigProb denominator = null;
				for (int t = 0; t < totalTime; t++) {
					if (arcProbs[i][j][t] != null) {
						if (denominator == null) {
							denominator = arcProbs[i][j][t];
						} else {
							denominator = denominator.add(arcProbs[i][j][t]);
						}
					}
				}
//				for (int component = 0; component < observationSize; component++) {
//					BigProb meanSum = null;
//					BigProb varSum = null;
//					for (int t = 0; t < totalTime; t++) {
//						if (arcProbs[i][j][t] != null) {
//							if (meanSum == null) {
//								meanSum = arcProbs[i][j][t].multiply(observationSeq[t][component]);
//							} else {
//								meanSum.add(arcProbs[i][j][t].multiply(observationSeq[t][component]));
//							}
//							if (varSum == null) {
//								varSum = arcProbs[i][j][t].multiply(Math.pow(observationSeq[t][component], 2));
//							} else {
//								varSum.add(arcProbs[i][j][t].multiply(Math.pow(observationSeq[t][component], 2)));
//							}
//						}
//					}
//					outputMeans[i][j][component] = meanSum.divideBy(denominator).toDouble();
//					outputVariances[i][j][component][component] = 
//							varSum.divideBy(denominator).toDouble() - Math.pow(outputMeans[i][j][component], 2);
//				}
				
				// System.out.println("Old mean(" + i + "," + j + "): " + Arrays.toString(outputMeans[i][j]));
				// for each vector component
				for (int component = 0; component < observationSize; component++) {
					double meanSum = 0;
					double varSum = 0;
					for (int t = 0; t < totalTime; t++) {
						if (arcProbs[i][j][t] != null) {
							double multiplier = arcProbs[i][j][t].divideBy(denominator).toDouble();
							if (multiplier < 0.00001) {
								multiplier = 0.00001;
							}
							meanSum += multiplier * observationSeq[t][component];
							varSum += multiplier * Math.pow(observationSeq[t][component], 2);
//							if (i == 0 && j == 0) {
//								System.out.println(t);
//								System.out.println(multiplier);
//								System.out.println(arcProbs[i][j][t]);
//								System.out.println("/" + denominator);
//								System.out.println(varSum);
//								System.out.println(observationSeq[t][component]);
//							}
//							if (varSum == 0) {
//								System.out.println(i + ", " + j + ", " + t);
//								System.out.println(multiplier);
//								System.out.println(observationSeq[t][component]);
//								System.out.println(arcProbs[i][j][t]);
//							}
						}
					}
					outputMeans[i][j][component] = meanSum;
					outputVariances[i][j][component][component] = varSum - Math.pow(meanSum, 2);
					// System.out.println("New var(" + i + "," + j + "," + component + "): " + outputVariances[i][j][component][component]);
				}
				// System.out.println("New mean(" + i + "," + j + "): " + Arrays.toString(outputMeans[i][j]));
			}
		}
	}
	
	@Override
	protected BigDecimal computeOutputProbability(int state1, int state2, int timeStep) {
		return Gaussian.multiGaussian(outputMeans[state1][state2], outputVariances[state1][state2], currentObservationSeq[timeStep]);
	}

	private void printModelParams(boolean trans, boolean means, boolean vars) {
		if (trans) {
			for (int i = 0; i < numStates; i++) {
				for (int j = i; j <= i+1 && j < numStates; j++) {
					System.out.println("a(" + i + "," + j + "): " + transitionProbs[i][j]);
				}
			}
		}
		if (means) {
			for (int i = 0; i < numStates; i++) {
				for (int j = i; j <= i+1 && j < numStates; j++) {
					System.out.println("mean(" + i + "," + j + "): ");
					for (int comp = 0; comp < observationSize; comp++) {
						System.out.print(outputMeans[i][j][comp] + " ");
					}
					System.out.println();
				}
			}
		}
	}
	
	/**************************************************
	 * BigDecimal Implementations
	 **************************************************/

	//		private BigDecimal[][] computeForwardProbabilities(double[][] observationSeq) {
	//			final int totalTime = observationSeq.length;
	//			BigDecimal[][] forwardProbs = new BigDecimal[totalTime + 1][numStates];
	//			// initialize the base case at state #1
	//			forwardProbs[0][0] = new BigDecimal("1");
	//			// for every timestep t
	//			for (int t = 1; t < totalTime + 1; t++) {
	//			// for (int t = 1; t < 10; t++) {
	//				// for each state s
	//				for (int s = 0; s < numStates; s++) {
	//					BigDecimal sum = null;
	//					// for each state that leads to this one i
	//					if (s > 0 && forwardProbs[t-1][s-1] != null) {
	//						sum = forwardProbs[t-1][s-1]
	//								.multiply(new BigDecimal(transitionProbs[s-1][s]))
	//								.multiply(Gaussian.multiGaussian(outputMeans[s-1][s], outputVariances[s-1][s], observationSeq[t-1]));
	//					}
	//					if (forwardProbs[t-1][s] != null) {
	//						BigDecimal secondComponent = forwardProbs[t-1][s]
	//						.multiply(new BigDecimal(transitionProbs[s][s]))
	//						.multiply(Gaussian.multiGaussian(outputMeans[s][s], outputVariances[s][s], observationSeq[t-1]));
	//						if (sum == null) {
	//							sum = secondComponent;
	//						}
	//						sum = sum.add(secondComponent);
	//					}
	//					System.out.println("Forward probability (" + t + "," + s + "): " + PrecisionMathUtils.bigDecimalToString(sum));
	//					forwardProbs[t][s] = sum;
	//				}
	//			}
	//			return forwardProbs;
	//		}
	
	//		private BigDecimal[][] computeBackwardProbabilities(double[][] observationSeq) {
	//			final int totalTime = observationSeq.length;
	//			BigDecimal[][] backwardProbs = new BigDecimal[totalTime + 1][numStates];
	//			// initialize the base case at state #1
	//			backwardProbs[totalTime][numStates-1] = new BigDecimal("1");
	//			// for every timestep t
	//			for (int t = totalTime-1; t >= 0; t--) {
	//				// for each state s
	//				for (int s = numStates-1; s >= 0; s--) {
	//					BigDecimal sum = null;
	//					// for each state that leads to this one i
	//					if (s < numStates-1 && backwardProbs[t+1][s+1] != null) {
	//						sum = backwardProbs[t+1][s+1]
	//								.multiply(new BigDecimal(transitionProbs[s][s+1]))
	//								.multiply(Gaussian.multiGaussian(outputMeans[s][s+1], outputVariances[s][s+1], observationSeq[t]));
	//					}
	//					if (backwardProbs[t+1][s] != null) {
	//						BigDecimal secondComponent = backwardProbs[t+1][s]
	//								.multiply(new BigDecimal(transitionProbs[s][s]))
	//								.multiply(Gaussian.multiGaussian(outputMeans[s][s], outputVariances[s][s], observationSeq[t]));
	//						if (sum == null) {
	//							sum = secondComponent;
	//						} else {
	//							sum = sum.add(secondComponent);
	//						}
	//					}
	//					System.out.println("Backward probability (" + t + "," + s + "): " + PrecisionMathUtils.bigDecimalToString(sum));
	//					backwardProbs[t][s] = sum;
	//				}
	//			}
	//			return backwardProbs;
	//		}
	
//	private BigDecimal[][][] computeArcProbabilities(BigDecimal[][] forwardProbs, BigDecimal[][] backwardProbs, double[][] observationSeq) {
//			final int totalTime = forwardProbs.length;
//			BigDecimal[][][] arcProbs = new BigDecimal[numStates][numStates][totalTime-1];
//			BigDecimal modelProbability = forwardProbs[totalTime-1][numStates-1];
//			// for every timestep t
//			for (int t = 0; t < totalTime - 1; t++) {
//				// for each state s
//				for (int s = 0; s < numStates; s++) {
//					// for each state that this one leads to
//					if (forwardProbs[t][s] != null && backwardProbs[t+1][s] != null) {
//						BigDecimal selfArcProb = forwardProbs[t][s]
//								.multiply(new BigDecimal(transitionProbs[s][s]))
//								.multiply(Gaussian.multiGaussian(outputMeans[s][s], outputVariances[s][s], observationSeq[t]))
//								.multiply(backwardProbs[t+1][s]);
//						System.out.println("Arc probability (" + s + ", " + s + ", " + t + "): " + PrecisionMathUtils.bigDecimalToString(selfArcProb));
//						arcProbs[s][s][t] = selfArcProb;
//					}
//					if (s < numStates-1 && forwardProbs[t][s+1] != null && backwardProbs[t+1][s+1] != null) {
//						BigDecimal nextStateArcProb = forwardProbs[t][s+1]
//								.multiply(new BigDecimal(transitionProbs[s][s+1]))
//								.multiply(Gaussian.multiGaussian(outputMeans[s][s+1], outputVariances[s][s+1], observationSeq[t]))
//								.multiply(backwardProbs[t+1][s+1]);
//						System.out.println("Arc probability (" + s + ", " + (s+1) + ", " + t + "): " + PrecisionMathUtils.bigDecimalToString(nextStateArcProb));
//						arcProbs[s][s+1][t] = nextStateArcProb;
//					}
//				}
//			}
//			return arcProbs;
//		}

	
	//		public void updateTransitionProbabilities(BigDecimal[][][] arcProbs) {
	//			final int totalTime = arcProbs[0][0].length;
	//			for (int i = 0; i < numStates; i++) {
	//				for (int j = i; j <= i+1 && j < numStates; j++) {
	//					BigDecimal numerator = null;
	//					for (int t = 0; t < totalTime; t++) {
	//						if (arcProbs[i][j][t] != null) {
	//							if (numerator == null) {
	//								numerator = arcProbs[i][j][t];
	//							} else {
	//								numerator = numerator.add(arcProbs[i][j][t]);
	//							}
	//						}
	//					}
	//					BigDecimal denominator = null;
	//					for (int n = 0; n < numStates; n++) {
	//						for (int t = 0; t < totalTime; t++) {
	//							if (arcProbs[i][n][t] != null) {
	//								if (denominator == null) {
	//									denominator = arcProbs[i][n][t];
	//								} else {
	//									denominator = denominator.add(arcProbs[i][n][t]);
	//								}
	//							}
	//						}
	//					}
	//					BigDecimal result = numerator.divide(denominator);
	//					System.out.print("Old a(" + i + "," + j + "): " + transitionProbs[i][j] + "   ");
	//					transitionProbs[i][j] = result.doubleValue();
	//					System.out.println("New a(" + i + "," + j + "): " + transitionProbs[i][j]);
	//				}
	//			}
	//		}
	
	//		public void updateMeansAndVariances(BigDecimal[][][] arcProbs, double[][] observationSeq) {
	//			final int totalTime = arcProbs[0][0].length;
	//			for (int i = 0; i < numStates; i++) {
	//				for (int j = 0; j < numStates; j++) {
	//					// first calculate the denominator
	//					BigDecimal denominator = null;
	//					for (int t = 0; t < totalTime; t++) {
	//						if (arcProbs[i][j][t] != null) {
	//							if (denominator == null) {
	//								denominator = arcProbs[i][j][t];
	//							} else {
	//								denominator = denominator.add(arcProbs[i][j][t]);
	//							}
	//						}
	//					}
	//					// for each vector component
	//					for (int component = 0; component < observationSize; component++) {
	//						double meanSum = 0;
	//						double varSum = 0;
	//						for (int t = 0; t < totalTime; t++) {
	//							if (arcProbs[i][j][t] != null) {
	//								double multiplier = arcProbs[i][j][t].divide(denominator).doubleValue();
	//								meanSum += multiplier * observationSeq[t][component];
	//								varSum += multiplier * Math.pow(observationSeq[t][component], 2);
	//							}
	//						}
	//						System.out.println("Old mean(" + i + "," + j + "): " + Arrays.toString(outputMeans[i][j]));
	//						outputMeans[i][j][component] = meanSum;
	//						System.out.println("New mean(" + i + "," + j + "): " + Arrays.toString(outputMeans[i][j]));
	//						outputVariances[i][j][component][component] = varSum - Math.pow(meanSum, 2);
	//					}
	//				}
	//			}
	//		}

	//		private BigDecimal computeMultiGaussianProb(double[] mean, double[][] variance, double[] obs) {
	//			BigDecimalWrapper[][] diff = new BigDecimalWrapper[mean.length][1];
	//			for (int i = 0; i < mean.length; i++) {
	//				diff[i][0] = DECIMAL_FACTORY.get(Math.abs(obs[i] - mean[i]));
	//			}
	//			BigDecimalWrapper[][] var = new BigDecimalWrapper[variance.length][variance.length];
	//			for (int i = 0; i < variance.length; i++) {
	//				for (int j = 0; j < variance[0].length; j++) {
	//					var[i][j] = DECIMAL_FACTORY.get(variance[i][j]);
	//				}
	//			}
	//			
	//			Matrix<BigDecimalWrapper> diffMatrix = new Matrix<>(diff);
	//			Matrix<BigDecimalWrapper> varMatrix = new Matrix<>(var);
	//			
	//			BigDecimalWrapper exponent = DECIMAL_FACTORY.get(-1/2.0)
	//					.multiply((diffMatrix.transpose().multiply(varMatrix.inverse()).multiply(diffMatrix)).getEntries()[0][0]);
	//			System.out.println(exponent.getValue().toString());
	//			double varianceDet = 1;
	//			for (int i = 0; i < variance.length; i++) {
	//				varianceDet *= variance[i][i];
	//			}
	//			System.out.println("|Var[][]| = " + varianceDet);
	//			// BigDecimalWrapper multiplier = DECIMAL_FACTORY.get(1/Math.sqrt(Math.pow(2*Math.PI, mean.length))).multiply(varMatrix.det());
	//			BigDecimal multiplier = new BigDecimal(1/(Math.sqrt(Math.pow(2*Math.PI, mean.length)) * varianceDet));
	//			System.out.println(multiplier.toString());
	//			BigDecimal eToTheExponent = PrecisionMathUtils.exp(exponent.getValue(), BIG_DECIMAL_PRECISION);
	//			BigDecimal result =  multiplier.multiply(eToTheExponent);
	//			System.out.println(result.toString());
	//			return result;

	//			Matrix diffMatrix = new Matrix(diff);
	//			Matrix varMatrix = new Matrix(variance);
	//			double exp = (-1/2.0)*(diffMatrix.transpose().times(varMatrix.inverse()).times(diffMatrix)).get(0,0);
	//			double multiplier = 1/Math.sqrt(Math.pow(2*Math.PI, mean.length)*varMatrix.det());
	//			double result = multiplier*Math.exp(exp);
	//			//System.out.println("Observation: " + Arrays.toString(obs));
	//			// System.out.println("Gaussian prob: " + result);
	//			return result;
	//		}
}
