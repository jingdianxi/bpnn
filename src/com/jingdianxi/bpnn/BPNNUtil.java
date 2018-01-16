package com.jingdianxi.bpnn;

import java.io.*;
import org.joone.engine.*;
import org.joone.engine.learning.TeachingSynapse;
import org.joone.io.*;
import org.joone.net.*;

/**
 * BPNN类，实现神经网络初始化、训练、测试等方法
 * 输入为0~1之间的double数（组）
 * 输出为0~1之间的double数（组）
 * 使用的接口方法可自定义执行内容
 */
public class BPNNUtil implements NeuralNetListener, Serializable {
	/**
	 * 序列号用于检查兼容性
	 */
	private static final long serialVersionUID = 1L;
	/**
	 * 神经网络成员
	 */
	private NeuralNet nnet = null;
	/**
	 * 神经网络保存路径
	 */
	private String nnetPath = null;

	/**
	 * 初始化神经网络
	 * @param nnetPath 神经网络存放路径
	 * @param inputNum 输入层神经元个数
	 * @param hiddenNum 隐藏层神经元个数
	 * @param outputNum 输出层神经元个数
	 */
	public void initBPNN(String nnetPath, int inputNum, int hiddenNum, int outputNum) {
		/* 设置新网络的保存路径 */
		this.nnetPath = nnetPath;
		/* 新建三个Layer，分别作为输入层，隐藏层，输出层 */
		LinearLayer input = new LinearLayer();
		SigmoidLayer hidden = new SigmoidLayer();
		SigmoidLayer output = new SigmoidLayer();
		/* 设置每个Layer包含的神经元个数 */
		input.setRows(inputNum);
		hidden.setRows(hiddenNum);
		output.setRows(outputNum);
		/* 新建两条突触，用于连接各层 */
		FullSynapse synapseIH = new FullSynapse();
		FullSynapse synapseHO = new FullSynapse();
		/* 连接输入-隐藏，隐藏-输出各层 */
		input.addOutputSynapse(synapseIH);
		hidden.addInputSynapse(synapseIH);
		hidden.addOutputSynapse(synapseHO);
		output.addInputSynapse(synapseHO);
		/* 新建一个神经网络，并添加输入层，隐藏层，输出层 */
		nnet = new NeuralNet();
		nnet.addLayer(input, NeuralNet.INPUT_LAYER);
		nnet.addLayer(hidden, NeuralNet.HIDDEN_LAYER);
		nnet.addLayer(output, NeuralNet.OUTPUT_LAYER);
	}

	/**
	 * 训练神经网络，使用磁盘文件，需搭配initBPNN
	 * @param trainFile 训练文件存放路径
	 * @param trainLength 训练文件行数
	 * @param rate 神经网络训练速度
	 * @param momentum 神经网络训练动量
	 * @param trainCicles 神经网络训练次数
	 */
	public void trainBPNN(String trainFile, int trainLength, double rate, double momentum, int trainCicles) {
		/* 获取输入层 */
		Layer input = nnet.getInputLayer();
		/* 新建输入突触 */
		FileInputSynapse trains = new FileInputSynapse();
		/* 设置输入文件 */
		trains.setInputFile(new File(trainFile));
		/* 设置使用的列数 */
		trains.setAdvancedColumnSelector("1,2,3,4");

		/* 获取输出层 */
		Layer output = nnet.getOutputLayer();
		/* 新建输入突触 */
		FileInputSynapse target = new FileInputSynapse();
		/* 设置输入文件 */
		target.setInputFile(new File(trainFile));
		/* 设置使用的列数 */
		target.setAdvancedColumnSelector("5,6,7,8");

		/* 新建训练突触 */
		TeachingSynapse trainer = new TeachingSynapse();
		/* 设置训练目标 */
		trainer.setDesired(target);

		/* 添加输入层的输入突触 */
		input.addInputSynapse(trains);
		/* 添加输出层的输出突触 */
		output.addOutputSynapse(trainer);
		/* 设置神经网络的训练突触 */
		nnet.setTeacher(trainer);

		/* 获取神经网络的监视器 */
		Monitor monitor = nnet.getMonitor();
		/* 设置训练速率 */
		monitor.setLearningRate(rate);
		/* 设置训练动量 */
		monitor.setMomentum(momentum);
		/* 新增监听者 */
		monitor.addNeuralNetListener(this);
		/* 设置训练数据个数（行数） */
		monitor.setTrainingPatterns(trainLength);
		/* 设置训练次数 */
		monitor.setTotCicles(trainCicles);
		/* 打开训练模式 */
		monitor.setLearning(true);
		/* 开始训练 */
		nnet.go();
	}

	/**
	 * 训练神经网络，使用内存数据，需搭配initBPNN
	 * @param TrainData 训练数据数组
	 * @param Rate 神经网络训练速度
	 * @param Momentum 神经网络训练动量
	 * @param TrainCicles 神经网络训练次数
	 */
	public void trainBPNN(double[][] trainData, double rate, double momentum, int trainCicles) {
		/* 设置输入数组 */
		Layer input = nnet.getInputLayer();
		MemoryInputSynapse trains = new MemoryInputSynapse();
		trains.setInputArray(trainData);
		trains.setAdvancedColumnSelector("1,2,3,4");

		/* 设置输出数组 */
		Layer output = nnet.getOutputLayer();
		MemoryInputSynapse target = new MemoryInputSynapse();
		target.setInputArray(trainData);
		target.setAdvancedColumnSelector("5,6,7,8");

		TeachingSynapse trainer = new TeachingSynapse();
		trainer.setDesired(target);
		input.addInputSynapse(trains);
		output.addOutputSynapse(trainer);
		nnet.setTeacher(trainer);

		Monitor monitor = nnet.getMonitor();
		monitor.setLearningRate(rate);
		monitor.setMomentum(momentum);
		monitor.addNeuralNetListener(this);
		monitor.setTrainingPatterns(trainData.length);
		monitor.setTotCicles(trainCicles);
		monitor.setLearning(true);
		nnet.go();
	}

	/**
	 * 训练已有神经网络，使用磁盘文件
	 * @param nnetPath 神经网络存放路径
	 * @param trainFile 训练文件存放路径
	 * @param trainLength 训练文件行数
	 * @param rate 神经网络训练速度
	 * @param momentum 神经网络训练动量
	 * @param trainCicles 神经网络训练次数
	 */
	public void trainBPNN(String nnetPath, String trainFile, int trainLength, double rate, double momentum,	int trainCicles) {
		// 初始化神经网络的保存路径
		this.nnetPath = nnetPath;
		// 获取保存的神经网络
		this.nnet = this.getBPNN(nnetPath);

		Monitor monitor = this.nnet.getMonitor();
		monitor.setLearningRate(rate);
		monitor.setMomentum(momentum);
		monitor.addNeuralNetListener(this);
		monitor.setTrainingPatterns(trainLength);
		monitor.setTotCicles(trainCicles);
		monitor.setLearning(true);
		this.nnet.go();
	}

	/**
	 * 测过已训练神经网络，使用磁盘文件
	 * @param nnetPath 神经网络存放路径
	 * @param outFile 测试结果存放路径
	 * @param testFile 测试文件存放路径
	 * @param testLength 测试文件行数
	 */
	public void testBPNN(String nnetPath, String outFile, String testFile, int testLength) {
		NeuralNet testBPNN = this.getBPNN(nnetPath);
		if (testBPNN != null) {
			Layer input = testBPNN.getInputLayer();
			/* 输入测试文件 */
			FileInputSynapse inputStream = new FileInputSynapse();
			inputStream.setInputFile(new File(testFile));
			inputStream.setAdvancedColumnSelector("1,2,3,4");
			input.removeAllInputs();
			input.addInputSynapse(inputStream);

			Layer output = testBPNN.getOutputLayer();
			/* 设置输出突触 */
			FileOutputSynapse fileOutput = new FileOutputSynapse();
			/* 设置输出文件保存路径 */
			fileOutput.setFileName(outFile);
			output.addOutputSynapse(fileOutput);

			Monitor monitor = testBPNN.getMonitor();
			monitor.setTrainingPatterns(testLength);
			monitor.setTotCicles(1);
			/* 关闭训练模式 */
			monitor.setLearning(false);

			/* 开始测试 */
			testBPNN.go();
			System.out.println("test");
		}
	}

	/**
	 * 测过已训练神经网络，使用矩阵数据
	 * @param nnetPath 神经网络存放路径
	 * @param outFile 测试结果存放路径
	 * @param testData 测试矩阵
	 */
	public void testBPNN(String nnetPath, String outFile, double[][] testData) {
		NeuralNet testBPNN = this.getBPNN(nnetPath);
		if (testBPNN != null) {
			Layer input = testBPNN.getInputLayer();
			/* 输入测试矩阵 */
			MemoryInputSynapse inputStream = new MemoryInputSynapse();
			input.removeAllInputs();
			input.addInputSynapse(inputStream);
			inputStream.setInputArray(testData);
			inputStream.setAdvancedColumnSelector("1,2,3,4");

			Layer output = testBPNN.getOutputLayer();
			FileOutputSynapse fileOutput = new FileOutputSynapse();
			fileOutput.setFileName(outFile);
			output.addOutputSynapse(fileOutput);

			Monitor monitor = testBPNN.getMonitor();
			monitor.setTrainingPatterns(testData.length);
			monitor.setTotCicles(1);
			monitor.setLearning(false);

			testBPNN.go();
			System.out.println("test");
		}
	}

	/**
	 * 测过已训练神经网络，使用内存矩阵
	 * @param nnetPath 神经网络存放路径
	 * @param testData 测试矩阵
	 * @return 测试结果
	 */
	public int[][] testBPNN(String nnetPath, double[][] testData) {
		NeuralNet testBPNN = this.getBPNN(nnetPath);
		int[][] result = new int[testData.length][2];
		if (testBPNN != null) {
			double[] temp = new double[2];

			Layer input = testBPNN.getInputLayer();
			/* 输入测试矩阵 */
			MemoryInputSynapse inputStream = new MemoryInputSynapse();
			input.removeAllInputs();
			input.addInputSynapse(inputStream);
			inputStream.setInputArray(testData);
			inputStream.setAdvancedColumnSelector("1,2,3,4");

			Layer output = testBPNN.getOutputLayer();
			MemoryOutputSynapse fileOutput = new MemoryOutputSynapse();
			output.addOutputSynapse(fileOutput);

			Monitor monitor = testBPNN.getMonitor();
			monitor.setTrainingPatterns(testData.length);
			monitor.setTotCicles(1);
			monitor.setLearning(false);
			testBPNN.go();

			for (int i = 0; i < result.length; i++) {
				temp = fileOutput.getNextPattern();
				result[i][0] = temp[0] < 0.5 ? 0 : 1;
				result[i][1] = temp[1] < 0.5 ? 0 : 1;
			}

			System.out.println("test");
			return result;
		}
		return result;
	}

	/**
	 * 读取已有神经网络
	 * @param nnetPath 神经网络存放路径
	 * @return
	 */
	NeuralNet getBPNN(String nnetPath) {
		NeuralNetLoader loader = new NeuralNetLoader(nnetPath);
		NeuralNet nnet = loader.getNeuralNet();
		return nnet;
	}

	/**
	 * 实现接口的方法
	 */
	@Override
	public void cicleTerminated(NeuralNetEvent event) {
		/* 获取监视器 */
		Monitor mon = (Monitor) event.getSource();
		/* 获取总训练次数 */
		long totalcicles = mon.getTotCicles();
		/* 获取当前训练次数 */
		long currentcicle = mon.getCurrentCicle();

		/* 调整训练速率 */
		if (currentcicle == (int) (totalcicles * 0.3)) {
			double rate = mon.getLearningRate();
			mon.setLearningRate(rate * 0.5);
		} else if (currentcicle == (int) (totalcicles * 0.5)) {
			double rate = mon.getLearningRate();
			mon.setLearningRate(rate * 0.5);
		} else if (currentcicle == (int) (totalcicles * 0.8)) {
			double rate = mon.getLearningRate();
			mon.setLearningRate(rate * 0.5);
		}

		/* 获取误差并输出 */
		double err = mon.getGlobalError();
		if (currentcicle % 10000 == 0) {
			System.out.println(currentcicle + " epochs remaining - RMSE = " + err);
		}
	}

	@Override
	public void netStarted(NeuralNetEvent event) {
		System.out.println("start");
	}

	@Override
	public void netStopped(NeuralNetEvent event) {
		System.out.println("Training Stopped...");
		Monitor mon = (Monitor) event.getSource();
		double err = mon.getGlobalError();
		System.out.println("Final - RMSE = " + err);
		/* 保存已训练网络 */
		try {
			FileOutputStream stream = new FileOutputStream(nnetPath);
			ObjectOutputStream out = new ObjectOutputStream(stream);
			/* 写入nnet对象 */
			out.writeObject(nnet);
			out.close();
			System.out.println("Save in " + nnetPath);
		} catch (Exception exception) {
			exception.printStackTrace();
		}
		/* 保存输出层结果 */
		NeuralNet nnet = this.getBPNN(nnetPath);
		if (nnet != null) {
			/* get the output layer */
			Layer output = nnet.getOutputLayer();
			/* create an output synapse */
			FileOutputSynapse fileOutput = new FileOutputSynapse();
			long currentTime = System.currentTimeMillis();
			fileOutput.setFileName("res/train_output_" + currentTime + ".txt");
			/* attach the output synapse to the last layer of the NN */
			output.addOutputSynapse(fileOutput);
			// Run the neural network only once (1 cycle) in recall mode
			nnet.getMonitor().setTotCicles(1);
			nnet.getMonitor().setLearning(false);
			nnet.go();
		}
	}

	@Override
	public void errorChanged(NeuralNetEvent event) {

	}

	@Override
	public void netStoppedError(NeuralNetEvent event, String error) {
		System.out.println("Network stopped due the following error: " + error);
	}

}