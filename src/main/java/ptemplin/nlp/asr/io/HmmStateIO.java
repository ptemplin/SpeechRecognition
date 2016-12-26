package ptemplin.nlp.asr.io;

import java.io.IOException;
import java.io.PrintWriter;

import ptemplin.nlp.asr.acoustic.ContinuousHMM;

public class HmmStateIO {
	
	public static final String OUTPUT_DIR = "tmp/";

	public static void saveFullHMMState(String modelName, ContinuousHMM model) {
		saveHmmTransitionProbs(modelName, model);
		saveHmmOutputParameters(modelName, model);
	}
	
	public static void saveHmmTransitionProbs(String modelName, ContinuousHMM model) {
		String fileName = OUTPUT_DIR + modelName + "_TP.mat";
		try (PrintWriter writer = new PrintWriter(fileName)){
			double[][] transitionProbs = model.getTransitionProbs();
			for (int i = 0; i < transitionProbs.length; i++) {
				for (int j = 0; j < transitionProbs[0].length; j++) {
					if (j != 0) {
						writer.print(" ");
					}
					writer.print(transitionProbs[i][j]);
				}
				writer.println();
			}
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}
	
	public static void saveHmmOutputParameters(String modelName, ContinuousHMM model) {
		String meanFileName = OUTPUT_DIR + modelName + "_MEAN.mat";
		String varFileName = OUTPUT_DIR + modelName + "_VAR.mat";
		try (PrintWriter writer = new PrintWriter(meanFileName)){
			double[][][] outputMeans = model.getOutputMeans();
			for (int i = 0; i < outputMeans.length; i++) {
				for (int j = 0; j < outputMeans[0].length; j++) {
					for (int k = 0; k < outputMeans[0][0].length; k++) {
						if (k != 0) {
							writer.print(" ");
						}
						writer.print(outputMeans[i][j][k]);
					}
					writer.println();
				}
			}
		} catch (IOException ex) {
			ex.printStackTrace();
		}
		try (PrintWriter writer = new PrintWriter(varFileName)){
			double[][][][] outputVars = model.getOutputVariances();
			for (int i = 0; i < outputVars.length; i++) {
				for (int j = 0; j < outputVars[0].length; j++) {
					for (int k = 0; k < outputVars[0][0].length; k++) {
						if (k != 0) {
							writer.print(" ");
						}
						writer.print(outputVars[i][j][k][k]);
					}
					writer.println();
				}
			}
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}
	
}
