package ptemplin.nlp.asr.acoustic;

import java.math.BigDecimal;

import ptemplin.nlp.asr.frontend.VectorQuantizer;
import ptemplin.nlp.asr.util.BigProb;

public class DiscreteHMM extends HiddenMarkovModel {
		
	private static final int NUM_CLASSES = 10;
	private static final double DEFAULT_OUTPUT_PROB = 1.0/NUM_CLASSES;
	
	private static final double MIN_OUTPUT_PROBABILITY = 0.000000001;

	private final double[][][] outputProbabilities;
	private final VectorQuantizer vectorQuantizer;
	
	private int[] currentObsQuant;

	public DiscreteHMM(VectorQuantizer vq) {
		super();
		outputProbabilities = new double[numStates][numStates][NUM_CLASSES];
		vectorQuantizer = vq;
		flatInitialize();
	}
	
	@Override
	public BigProb evaluateObservation(double[][] observationSeq) {
		quantizeObservation(observationSeq);
		return super.evaluateObservation(observationSeq);
	}
	
	@Override
	public void train(double[][] observationSeq) {
		quantizeObservation(observationSeq);
		super.train(observationSeq);
	}
	
	private void quantizeObservation(double[][] observationSeq) {
		currentObsQuant = new int[observationSeq.length];
		for (int i = 0; i < observationSeq.length; i++) {
			currentObsQuant[i] = vectorQuantizer.quantizeObservation(observationSeq[i]);
		}
	}

	private void flatInitialize() {
		// initialize transition probabilities equally
		for (int i = 0; i < numStates-1; i++) {
			for (int j = i; j <= i+1; j++) {
				transitionProbs[i][j] = 1.0/2;
			}
		}
		transitionProbs[numStates-1][numStates-1] = 1;
		// initialize output probabilities equally
		for (int i = 0; i < numStates; i++) {
			for (int j = 0; j < numStates; j++) {
				for (int k = 0; k < NUM_CLASSES; k++) {
					outputProbabilities[i][j][k] = DEFAULT_OUTPUT_PROB;
				}
			}
		}
	}

	@Override
	public void updateOutputParameters(BigProb[][][] arcProbs, double[][] observationSeq) {
		final int totalTime = arcProbs[0][0].length;
		for (int i = 0; i < numStates; i++) {
			for (int j = 0; j < numStates; j++) {
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
				// for each possible output,
				for (int k = 0; k < NUM_CLASSES; k++) {
					BigProb numerator = null;
					for (int t = 0; t < totalTime; t++) {
						if (arcProbs[i][j][t] != null) {
							if (currentObsQuant[t] == k) {
								if (numerator == null) {
									numerator = arcProbs[i][j][t];
								} else {
									numerator = numerator.add(arcProbs[i][j][t]);
								}
							}
						}
					}
					if (numerator == null) {
						outputProbabilities[i][j][k] = MIN_OUTPUT_PROBABILITY;
						continue;
					}
					double outputProb = numerator.divideBy(denominator).toDouble();
					if (outputProb < MIN_OUTPUT_PROBABILITY) {
						outputProb = MIN_OUTPUT_PROBABILITY;
					}
					outputProbabilities[i][j][k] = outputProb;
					//System.out.println("New mean(" + i + "," + j + "): " + outputProbabilities[i][j][k]);
				}
			}
		}
	}
	
	@Override
	protected BigDecimal computeOutputProbability(int state1, int state2, int timeStep) {
		return new BigDecimal(outputProbabilities[state1][state2][currentObsQuant[timeStep]]);
	}
}
