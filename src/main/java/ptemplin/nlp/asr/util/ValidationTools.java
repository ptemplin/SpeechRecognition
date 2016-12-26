package ptemplin.nlp.asr.util;

public class ValidationTools {

	public static void validateArcProbabilities(BigProb[][][] arcProbs) {
		int numStates = arcProbs.length;
		int totalTime = arcProbs[0][0].length;
		printArcProbs(arcProbs);
		sumArcProbs(arcProbs);
	}
	
	private static void printArcProbs(BigProb[][][] arcProbs) {
		int numStates = arcProbs.length;
		int totalTime = arcProbs[0][0].length;
		for (int i = 0; i < numStates; i++) {
			for (int j = i; j <= i+1 && j < numStates; j++) {
				for (int t = 0; t < totalTime; t++) {
					System.out.println("arc(" + i + "," + j + "," + t + "): " + arcProbs[i][j][t]);
				}
			}
		}
	}
	
	private static void sumArcProbs(BigProb[][][] arcProbs) {
		int numStates = arcProbs.length;
		int totalTime = arcProbs[0][0].length;
		BigProb sum = null;
		for (int i = 0; i < numStates; i++) {
			for (int j = i; j <= i+1 && j < numStates; j++) {
				for (int t = 0; t < totalTime; t++) {
					if (arcProbs[i][j][t] != null) {
						if (sum == null) {
							sum = arcProbs[i][j][t];
						} else {
							sum.add(arcProbs[i][j][t]);
						}
					}
				}
			}
		}
		System.out.println("Sum: " + sum);
	}
	
}
