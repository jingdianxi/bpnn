package com.jingdianxi.bpnn;

public class BPNNTest {

	public static void main(String[] args) {
		String nnetPath = "res/nnet.snet";
		String trainFile = "res/train.txt";
		String testFile = "res/test.txt";
		String outputFile = "res/result.txt";

		/* double[][] testData = new double[][] {
			{ 0.5781, 0.7307, 0.6190, 0.2249, 0, 0, 0, 1 },
			{ 0.6250, 0.7557, 0.3185, 0.1388, 0, 0, 1, 0 },
			{ 0.5313, 0.7140, 0.8305, 0.3589, 0, 0, 0, 1 },
			{ 0.6719, 0.7724, 0.1923, 0.0813, 0, 0, 1, 0 },
			{ 0.5313, 0.7182, 0.6550, 0.4019, 0, 0, 1, 0 },
			{ 0.6094, 0.8205, 0.3690, 0.2392, 0, 1, 0, 0 },
			{ 0.4609, 0.6701, 0.7873, 0.4737, 0, 0, 0, 1 },
			{ 0.6484, 0.8810, 0.2452, 0.1818, 0, 1, 0, 0 },
			{ 0.2734, 0.6743, 0.8293, 0.5407, 0, 0, 1, 0 },
			{ 0.5859, 0.8664, 0.4772, 0.3206, 0, 1, 0, 0 },
			{ 0.2266, 0.6054, 1.0000, 0.7033, 0, 0, 0, 1 },
			{ 0.6406, 0.9582, 0.3101, 0.1962, 0, 1, 0, 0 }
		}; */
		BPNNUtil bpnnUtil = new BPNNUtil();
		bpnnUtil.initBPNN(nnetPath, 4, 11, 4);
		bpnnUtil.trainBPNN(trainFile, 48, 0.8, 0.3, 1000000);
		// int[][] result = bpnnUtil.testBPNN(nnetPath, testData);
		// bpnnUtil.trainBPNN(nnetPath, trainFile, 8032, 0.4, 0.3, 5000);
		// bpnnUtil.trainBPNN(trainData, 0.8, 0.3, 1000);
		bpnnUtil.testBPNN(nnetPath, outputFile, testFile, 12);
		// bpnnUtil.testBPNN(nnetPath, outputFile, testData);
		// int[][] result = bpnnUtil.testBPNN(nnetPath, testData);
	}

}
