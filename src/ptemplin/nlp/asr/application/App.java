package ptemplin.nlp.asr.application;

import java.util.List;

import ptemplin.nlp.asr.acoustic.ContinuousHMM;
import ptemplin.nlp.asr.acoustic.HiddenMarkovModel;
import ptemplin.nlp.asr.frontend.LincolnFrontend;
import ptemplin.nlp.asr.frontend.SilenceProcessor;
import ptemplin.nlp.asr.io.HmmLoadSave;
import ptemplin.nlp.asr.io.SpeechFileReader;
import ptemplin.nlp.asr.io.SpeechSample;
import ptemplin.nlp.asr.util.BigProb;

public class App {
	
	private static final LincolnFrontend analyzer = new LincolnFrontend();
	private static final SpeechFileReader reader = new SpeechFileReader();
	
	private static final String[] vocabulary = {"one", "two", "three"};
	private static final int NUM_TRAINING_SAMPLES = 10;

	public static void main(String[] args) throws Exception {
		initHMMAndMultiTrain();
	}
	
	// Test training analyzed sample against model
	public static void initHmmAndTrain() throws Exception {
		System.out.println("Initializing HMM");
		HiddenMarkovModel acousticModel = new ContinuousHMM();

		System.out.println("Extracting features from test data");
		SpeechSample sample = reader.readSpeechFileForTrainingData("one", 1);
		int[] trimmedSpeech = SilenceProcessor.trimSilence(sample.getDataAsInts());
		List<int[]> observationSeq = analyzer.computeObservationVectors(trimmedSpeech);
		
		System.out.println("Testing data against default model");
		BigProb likelihood = acousticModel.evaluateObservation(observationSeq);
		System.out.println("P(O|M) = " + likelihood);
		
		System.out.println("Training model based on data");
		acousticModel.train(observationSeq);
		
		System.out.println("Testing data against 1xtrained model");
		likelihood = acousticModel.evaluateObservation(observationSeq);
		System.out.println("P(O|M') = " + likelihood);
	}
	
	// Test multi-training analyzed sample against model
	public static void initHMMAndMultiTrain() throws Exception {
		System.out.println("Initializing HMM");
		HiddenMarkovModel acousticModel = new ContinuousHMM();

		System.out.println("Extracting features from test data");
		SpeechSample sample = reader.readSpeechFileForTrainingData("one", 2);
		int[] trimmedSpeech = SilenceProcessor.trimSilence(sample.getDataAsInts());
		List<int[]> observationSeq = analyzer.computeObservationVectors(trimmedSpeech);
		printObservationVector(observationSeq);

		System.out.println("Testing data against default model");
		BigProb likelihood = acousticModel.evaluateObservation(observationSeq);
		System.out.println("P(O|M) = " + likelihood);

		System.out.println("Training model based on data");
		acousticModel.train(observationSeq);
		HmmLoadSave.saveFullHMMState("1xTrained_MultiTrain", (ContinuousHMM) acousticModel);

		System.out.println("Testing data against 1xtrained model");
		likelihood = acousticModel.evaluateObservation(observationSeq);
		System.out.println("P(O|M') = " + likelihood);
		
		System.out.println("Training model based on data");
		acousticModel.train(observationSeq);
		HmmLoadSave.saveFullHMMState("2xTrained_MultiTrain", (ContinuousHMM) acousticModel);

		System.out.println("Testing data against 2xtrained model");
		likelihood = acousticModel.evaluateObservation(observationSeq);
		System.out.println("P(O|M'') = " + likelihood);
		
		System.out.println("Training model based on data");
		acousticModel.train(observationSeq);
		HmmLoadSave.saveFullHMMState("3xTrained_MultiTrain", (ContinuousHMM) acousticModel);

		System.out.println("Testing data against 3xtrained model");
		likelihood = acousticModel.evaluateObservation(observationSeq);
		System.out.println("P(O|M''') = " + likelihood);
	}
	
	public static void trainFullModelAndEvaluateSamples() throws Exception {
		String trainingWord = "one";
			
		System.out.println("Initializing HMM");
		HiddenMarkovModel acousticModel = new ContinuousHMM();
		HmmLoadSave.saveFullHMMState("Default", (ContinuousHMM) acousticModel);

		System.out.println("Training on single word...");
		for (int i = 1; i <= 10; i++) {
			SpeechSample sample = reader.readSpeechFileForTrainingData(trainingWord, i);
			int[] trimmedSpeech = SilenceProcessor.trimSilence(sample.getDataAsInts());
			List<int[]> observationSeq = analyzer.computeObservationVectors(trimmedSpeech);
			acousticModel.train(observationSeq);
			HmmLoadSave.saveFullHMMState(i + "xTrained", (ContinuousHMM) acousticModel); 
		}
		
		System.out.println("Testing all vocabulary words against model...");
		for (String word : vocabulary) {
			if (word.equals(trainingWord)) {
				System.out.println("Training word (should be high)");
			} else { System.out.println("Non-trained word (should be low)"); }
			for (int i = 1; i <= 10; i++) {
				SpeechSample sample = reader.readSpeechFileForTrainingData(word, i);
				int[] trimmedSpeech = SilenceProcessor.trimSilence(sample.getDataAsInts());
				List<int[]> observationSeq = analyzer.computeObservationVectors(trimmedSpeech);
				BigProb likelihood = acousticModel.evaluateObservation(observationSeq);
				System.out.println("Word: " + word + " Sample: " + i + " P(O|M) = " + likelihood);
			}
		}
	}
	
	public static void trainAndEvaluate3Samples() throws Exception {
		System.out.println("Initializing HMM");
		HiddenMarkovModel acousticModel = new ContinuousHMM();

		System.out.println("Extracting features from test data");
		SpeechSample sample1 = reader.readSpeechFileForTrainingData("one", 1);
		int[] trimmedSpeech1 = SilenceProcessor.trimSilence(sample1.getDataAsInts());
		List<int[]> observationSeq1 = analyzer.computeObservationVectors(trimmedSpeech1);
		SpeechSample sample2 = reader.readSpeechFileForTrainingData("one", 2);
		int[] trimmedSpeech2 = SilenceProcessor.trimSilence(sample2.getDataAsInts());
		List<int[]> observationSeq2 = analyzer.computeObservationVectors(trimmedSpeech2);
		SpeechSample sample3 = reader.readSpeechFileForTrainingData("one", 3);
		int[] trimmedSpeech3 = SilenceProcessor.trimSilence(sample3.getDataAsInts());
		List<int[]> observationSeq3 = analyzer.computeObservationVectors(trimmedSpeech3);
		SpeechSample sample4 = reader.readSpeechFileForTrainingData("two", 5);
		int[] trimmedSpeech4 = SilenceProcessor.trimSilence(sample4.getDataAsInts());
		List<int[]> observationSeq4 = analyzer.computeObservationVectors(trimmedSpeech4);
		

		System.out.println("Testing data against default model");
		BigProb likelihood = acousticModel.evaluateObservation(observationSeq1);
		System.out.println("P(O_1|M) = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq2);
		System.out.println("P(O_2|M) = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq3);
		System.out.println("P(O_3|M) = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq4);
		System.out.println("P(O_4|M) = " + likelihood + "    (Not a one)");

		System.out.println("Training model based on sample data #1");
		acousticModel.train(observationSeq1);

		System.out.println("Testing data against 1xtrained model");
		likelihood = acousticModel.evaluateObservation(observationSeq1);
		System.out.println("P(O_1|M') = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq2);
		System.out.println("P(O_2|M') = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq3);
		System.out.println("P(O_3|M') = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq4);
		System.out.println("P(O_4|M) = " + likelihood + "    (Not a one)");
		
		System.out.println("Training model based on sample data #2");
		acousticModel.train(observationSeq2);

		System.out.println("Testing data against 2xtrained model");
		likelihood = acousticModel.evaluateObservation(observationSeq1);
		System.out.println("P(O_1|M'') = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq2);
		System.out.println("P(O_2|M'') = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq3);
		System.out.println("P(O_3|M'') = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq4);
		System.out.println("P(O_4|M) = " + likelihood + "    (Not a one)");
		
		System.out.println("Training model based on sample data #3");
		acousticModel.train(observationSeq3);

		System.out.println("Testing data against 3xtrained model");
		likelihood = acousticModel.evaluateObservation(observationSeq1);
		System.out.println("P(O_1|M''') = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq2);
		System.out.println("P(O_2|M''') = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq3);
		System.out.println("P(O_3|M''') = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq4);
		System.out.println("P(O_4|M) = " + likelihood + "    (Not a one)");
	}
	
	// Test feature analysis, observation evaluation against default model
	public static void initHmmAndEvaluteTestObservationSequence() throws Exception {
		System.out.println("Initializing default HMM");
		HiddenMarkovModel acousticModel = new ContinuousHMM();
		
		System.out.println("Extracting features from test data");
		SpeechSample sample = reader.readSpeechFileForTrainingData("one", 1);
		int[] trimmedSpeech = SilenceProcessor.trimSilence(sample.getDataAsInts());
		List<int[]> observationSeq = analyzer.computeObservationVectors(trimmedSpeech);
		
		System.out.println("Testing data against default model");
		BigProb likelihood = acousticModel.evaluateObservation(observationSeq);
		//System.out.println("P(O|M) = " + PrecisionMathUtils.bigDecimalToString(likelihood));
		System.out.println("P(O|M) = " + likelihood);
	}
	
	private static void printObservationVector(List<int[]> observationSequence) {
		for (int i = 0; i < observationSequence.get(0).length; i++) {
			for (int j = 0; j < observationSequence.size(); j++) {
				System.out.format("%-4d  ", observationSequence.get(j)[i]);
			}
			System.out.println();
		}
	}
	
}
