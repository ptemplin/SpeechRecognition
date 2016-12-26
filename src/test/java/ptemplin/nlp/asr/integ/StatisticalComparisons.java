package ptemplin.nlp.asr.integ;

import ptemplin.nlp.asr.frontend.SpectralAnalyzer;
import ptemplin.nlp.asr.frontend.lincoln.LincolnFrontend;
import ptemplin.nlp.asr.frontend.lincoln.SilenceProcessor;
import ptemplin.nlp.asr.io.SpeechFileReader;
import ptemplin.nlp.asr.io.SpeechSample;

import java.util.Arrays;
import java.util.List;

import static ptemplin.nlp.asr.frontend.FeatureAnalyzer.NUM_MFCC_COMPONENTS;

public class StatisticalComparisons {

    private static final SpectralAnalyzer analyzer = new LincolnFrontend();
    private static final SpeechFileReader reader = new SpeechFileReader();

    private static final String[] vocabulary = {"one", "two", "three"};
    private static final int NUM_TRAINING_SAMPLES = 10;
    private static final int OBSERVATION_SIZE = NUM_MFCC_COMPONENTS;

    public static void main(String[] args) throws Exception {
        computeStatsForWordClass("one");
        computeStatsForWordClass("two");
        computeStatsForWordClass("three");
    }

    public static void computeStatsForWordClass(String className) throws Exception {

        double[] classMeans = new double[OBSERVATION_SIZE];
        double[] classVars = new double[OBSERVATION_SIZE];
        for (int i = 1; i <= NUM_TRAINING_SAMPLES; i++) {
            //System.out.println("Extracting features from test data " + i);
            SpeechSample sample = reader.readSpeechFileForTrainingData(className, i);
            int[] trimmedSpeech = SilenceProcessor.trimSilence(sample.getDataAsInts());
            List<int[]> observationSeq = analyzer.computeFeatureVectors(trimmedSpeech);

            // get the mean for each component across the word
            double[] componentMeans = new double[OBSERVATION_SIZE];
            int totalSamples = observationSeq.size();
            for (int t = 0; t < totalSamples; t++) {
                for (int c = 0; c < OBSERVATION_SIZE; c++) {
                    componentMeans[c] += observationSeq.get(t)[c];
                }
            }
            for (int c = 0; c < componentMeans.length; c++) {
                componentMeans[c] /= totalSamples;
                classMeans[c] += componentMeans[c];
            }

            // get the variance for each component across the word
            double[] componentVars = new double[OBSERVATION_SIZE];
            for (int t = 0; t < totalSamples; t++) {
                for (int c = 0; c < OBSERVATION_SIZE; c++) {
                    componentVars[c] += Math.pow(observationSeq.get(t)[c] - componentMeans[c], 2);
                }
            }
            for (int c = 0; c < componentVars.length; c++) {
                componentVars[c] /= totalSamples;
                classVars[c] += componentVars[c];
            }

            //System.out.println(Arrays.toString(componentMeans));
            //System.out.println(Arrays.toString(componentVars));
        }
        for (int i = 0; i < OBSERVATION_SIZE; i++) {
            classMeans[i] /= NUM_TRAINING_SAMPLES;
            classVars[i] /= NUM_TRAINING_SAMPLES;
        }
        System.out.println("Means: " + Arrays.toString(classMeans));
        System.out.println("Variances: " + Arrays.toString(classVars));
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

