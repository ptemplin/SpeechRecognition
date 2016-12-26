package ptemplin.nlp.asr.acoustic;

import ptemplin.nlp.asr.util.Gaussian;

import java.math.BigDecimal;
import java.util.Arrays;

public class ContinuousHMM extends HiddenMarkovModel {

	private static final double[] DEFAULT_MEANS = {-20, 70, 30, 43, 26, 10, 12, 0, 4, 0, 15, 0};
    private static final double[] DEFAULT_VARIANCES = {1800, 850, 500, 450, 350, 300, 300, 250, 250, 215, 180, 150};

	private final double[][][] outputMeans;
	private final double[][][][] outputVariances;

	public ContinuousHMM() {
		super();
		outputMeans = new double[numStates][numStates][observationSize];
		outputVariances = new double[numStates][numStates][observationSize][observationSize];
		flatInitialize();
	}

	public double[][][] getOutputMeans() {
		return outputMeans;
	}

	public double[][][][] getOutputVariances() {
		return outputVariances;
	}

	private void flatInitialize() {
		// initialize transition probabilities equally
		for (int i = 0; i < numStates - 1; i++) {
			for (int j = i; j <= i + 1; j++) {
				transitionProbs[i][j] = 0.5d;
			}
		}
		transitionProbs[numStates - 1][numStates - 1] = 1.d;
		// initialize output means equally
		for (int i = 0; i < numStates; i++) {
			for (int j = i; j <= i + 1 && j < numStates; j++) {
				for (int k = 0; k < observationSize; k++) {
					outputMeans[i][j][k] = DEFAULT_MEANS[k];
				}
			}
		}
		// initialize variances equally
		for (int i = 0; i < numStates; i++) {
			for (int j = i; j <= i + 1 && j < numStates; j++) {
				for (int k = 0; k < observationSize; k++) {
					outputVariances[i][j][k][k] = DEFAULT_VARIANCES[k];
				}
			}
		}
	}

	// @Override
	public void updateOutputParameters2(double[][][] arcProbs, double[][] observationSeq) {
		final int totalTime = arcProbs[0][0].length;
		for (int i = 0; i < numStates; i++) {
			for (int j = i; j <= i + 1 && j < numStates; j++) {
				// first calculate the denominator
				double logDenominator = Double.NEGATIVE_INFINITY;
				boolean first = true;
				for (int t = 0; t < totalTime; t++) {
					if (arcProbs[i][j][t] != 0.d) {
						if (first) {
							logDenominator = arcProbs[i][j][t];
							first = false;
						} else {
							logDenominator = logMath.addAsLinear(logDenominator, arcProbs[i][j][t]);
						}
					}
				}
				System.out.println("Log denominator: " + logDenominator);
				// for each vector component
				for (int component = 0; component < observationSize; component++) {
					double logMeanSum = Double.NEGATIVE_INFINITY;
					double logVarSum = Double.NEGATIVE_INFINITY;
                    first = true;
					for (int t = 0; t < totalTime; t++) {
                        if (arcProbs[i][j][t] != 0.d) {
                            //System.out.println(arcProbs[i][j][t] + "-" + logDenominator);
                            double logMultiplier = arcProbs[i][j][t] - logDenominator;
                            //System.out.println("=" + logMultiplier);
                            if (logMultiplier < -100000) {
                                logMultiplier = -100000;
                            }
                            double logMeanPart = logMultiplier + logMath.linearToLog(observationSeq[t][component]);
                            double logVarPart = logMultiplier + logMath.linearToLog(Math.pow(observationSeq[t][component], 2));
                            if (first) {
                                logMeanSum = logMeanPart;
                                logVarSum = logVarPart;
                                first = false;
                            } else {
                                logMath.addAsLinear(logMeanSum, logMeanPart);
                                logMath.addAsLinear(logVarSum, logVarPart);
                            }
                        }
                    }
					outputMeans[i][j][component] = logMath.logToLinear(logMeanSum);
                    System.out.println(String.format("outputMeans[%d][%d][%d]=%.10f", i, j, component, outputMeans[i][j][component]));
					double logVarMeanDiff = logMath.subtractAsLinear(logVarSum, 2 * logMeanSum);
					outputVariances[i][j][component][component] = logMath.logToLinear(logVarMeanDiff);
                    System.out.println(String.format("outputVars[%d][%d][%d]=%.10f", i, j, component, outputVariances[i][j][component][component]));
				}
			}
		}
	}

	@Override
    public void updateOutputParameters(double[][][] logArcProbs, double[][] observationSeq) {
        final int totalTime = logArcProbs[0][0].length;
        for (int i = 0; i < numStates; i++) {
            for (int j = i; j <= i + 1 && j < numStates; j++) {
                // first calculate the denominator
                double logDenominator = Double.NEGATIVE_INFINITY;
                boolean first = true;
                for (int t = 0; t < totalTime; t++) {
                    if (logArcProbs[i][j][t] != 0.d) {
                        if (first) {
                            logDenominator = logArcProbs[i][j][t];
                            first = false;
                        } else {
                            logDenominator = logMath.addAsLinear(logDenominator, logArcProbs[i][j][t]);
                        }
                    }
                }
                double denominator = logMath.logToLinear(logDenominator);
                if (i == 3 && j == 4) {
                    // System.out.println("Old Parameters");
                    System.out.println("Denominator: " + denominator);
                    // printStateParams(i, j);
                }
                // for each vector component
                for (int component = 0; component < observationSize; component++) {
                    double meanSum = Double.NEGATIVE_INFINITY;
                    double varSum = Double.NEGATIVE_INFINITY;
                    first = true;
                    for (int t = 0; t < totalTime; t++) {
                        if (logArcProbs[i][j][t] != 0.d) {
                            double arcProb = logMath.logToLinear(logArcProbs[i][j][t]);
                            double meanPart = arcProb * observationSeq[t][component];
                            double varPart = arcProb * Math.pow(observationSeq[t][component] - outputMeans[i][j][component], 2);
                            if (first) {
                                meanSum = meanPart;
                                varSum = varPart;
                                first = false;
                            } else {
                                meanSum += meanPart;
                                varSum += varPart;
                            }
                        }
                    }
                    outputMeans[i][j][component] = meanSum / denominator;
                    outputVariances[i][j][component][component] = varSum / denominator;
                }
                if (i == 3 && j == 4) {
                    printStateParams(i, j);
                }
            }
        }
    }

	@Override
	protected double computeOutputProbability(int state1, int state2, int timeStep) {
		BigDecimal gaussianResult = Gaussian.multiGaussian(outputMeans[state1][state2],
				outputVariances[state1][state2],
				currentObservationSeq[timeStep]);
        //System.out.println("Linear output prob: " + gaussianResult.toString().substring(0,10));
        double doubleResult = gaussianResult.doubleValue();
        //printOutputProbabilitySummary(state1, state2, timeStep, doubleResult);
        if (doubleResult < 1e-300) {
            doubleResult = 1e-300;
        }
		return logMath.linearToLog(doubleResult);
	}

	private void printModelParams(boolean trans, boolean means, boolean vars) {
		if (trans) {
			for (int i = 0; i < numStates; i++) {
				for (int j = i; j <= i + 1 && j < numStates; j++) {
					System.out.println("a(" + i + "," + j + "): " + transitionProbs[i][j]);
				}
			}
		}
		if (means) {
			for (int i = 0; i < numStates; i++) {
				for (int j = i; j <= i + 1 && j < numStates; j++) {
					System.out.print("mean(" + i + "," + j + "): [");
					for (int comp = 0; comp < observationSize; comp++) {
						System.out.print(outputMeans[i][j][comp] + ", ");
					}
					System.out.println("]");
				}
			}
		}
		if (vars) {
            for (int i = 0; i < numStates; i++) {
                for (int j = i; j <= i + 1 && j < numStates; j++) {
                    System.out.print("var(" + i + "," + j + "): [");
                    for (int comp = 0; comp < observationSize; comp++) {
                        System.out.print(outputVariances[i][j][comp][comp] + ", ");
                    }
                    System.out.println("]");
                }
            }
        }
	}

	private void printStateParams(int i, int j) {
        System.out.print("mean(" + i + "," + j + "): [");
        for (int comp = 0; comp < observationSize; comp++) {
            System.out.print(outputMeans[i][j][comp] + ", ");
        }
        System.out.println("]");

//        System.out.print("var(" + i + "," + j + "): [");
//        for (int comp = 0; comp < observationSize; comp++) {
//            System.out.print(outputVariances[i][j][comp][comp] + ", ");
//        }
//        System.out.println("]");

        System.out.print("sd(" + i + "," + j + "): [");
        for (int comp = 0; comp < observationSize; comp++) {
            System.out.print(Math.sqrt(outputVariances[i][j][comp][comp]) + ", ");
        }
        System.out.println("]");
    }

    private void printOutputProbabilitySummary(int state1, int state2, int timeStep, double result) {
        if (state1 == 2 && state2 == 3 && timeStep == 10) {
            printStateParams(state1, state2);
            System.out.print("#sd's(" + state1 + "," + state2 + "): [");
            double totalSds = 0;
            for (int comp = 0; comp < observationSize; comp++) {
                double meanDiff = currentObservationSeq[timeStep][comp] - outputMeans[state1][state2][comp];
                double sds = Math.abs(meanDiff / Math.sqrt(outputVariances[state1][state2][comp][comp]));
                totalSds += sds;
                System.out.print(sds + ", ");
            }
            System.out.println("]");
            System.out.println("Total sd's: " + totalSds);
            System.out.println("Observation: " + Arrays.toString(currentObservationSeq[timeStep]));
            System.out.println("Double output prob: " + result);
        }
    }
}