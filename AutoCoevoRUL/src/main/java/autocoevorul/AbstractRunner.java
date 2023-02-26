package autocoevorul;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.aeonbits.owner.ConfigFactory;
import org.apache.commons.io.FileUtils;
import org.api4.java.ai.ml.core.dataset.splitter.SplitFailedException;
import org.api4.java.ai.ml.core.evaluation.IPrediction;
import org.api4.java.ai.ml.core.evaluation.IPredictionBatch;
import org.api4.java.ai.ml.core.exception.PredictionException;
import org.api4.java.ai.ml.core.exception.TrainingException;
import org.api4.java.ai.ml.regression.evaluation.IRegressionPrediction;
import org.api4.java.algorithm.exceptions.AlgorithmExecutionCanceledException;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;
import org.slf4j.Logger;

import ai.libs.jaicore.components.exceptions.ComponentNotFoundException;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.experiments.ExperimentDatabasePreparer;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetConfig;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;
import ai.libs.jaicore.experiments.exceptions.ExperimentAlreadyExistsInDatabaseException;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.experiments.exceptions.IllegalExperimentSetupException;
import ai.libs.jaicore.ml.core.EScikitLearnProblemType;
import ai.libs.jaicore.ml.regression.loss.ERulPerformanceMeasure;
import ai.libs.jaicore.ml.regression.singlelabel.SingleTargetRegressionPrediction;
import ai.libs.jaicore.ml.scikitwrapper.ScikitLearnWrapper;
import autocoevorul.baseline.randomsearch.RandomSearchRunner;
import autocoevorul.experiment.ExperimentConfiguration;
import autocoevorul.experiment.ICoevolutionConfig;

public abstract class AbstractRunner {

	protected static final Logger LOGGER = org.slf4j.LoggerFactory.getLogger(RandomSearchRunner.class);

	public abstract String getExperimentConfigurationFilePath();

	public abstract String getDatabaseConfigurationFilePath();

	public void run(final String executor, final boolean setupDatabase, final boolean executeExperiments) throws ExperimentEvaluationFailedException, IOException, ComponentNotFoundException,
			ClassNotFoundException, InterruptedException, SplitFailedException, NoSuchFieldException, SecurityException, IllegalArgumentException, IllegalAccessException {

		ICoevolutionConfig expConfig = (ICoevolutionConfig) ConfigFactory.create(ICoevolutionConfig.class).loadPropertiesFromFile(new File(this.getExperimentConfigurationFilePath()));
		IDatabaseConfig dbConfig = (IDatabaseConfig) ConfigFactory.create(IDatabaseConfig.class).loadPropertiesFromFile(new File(this.getDatabaseConfigurationFilePath()));
		ExperimenterMySQLHandle handle = new ExperimenterMySQLHandle(dbConfig);

		if (setupDatabase) {
			this.setupDatabase(expConfig, handle);
		}
		if (executeExperiments) {
			this.executeExperiments(executor, expConfig, dbConfig, handle);
		}
	}

	private void setupDatabase(final IExperimentSetConfig expConfig, final ExperimenterMySQLHandle handle) {
		try {
			handle.setup(expConfig);
		} catch (ExperimentDBInteractionFailedException e) {
			LOGGER.error("Couldn't setup the sql handle.", e);
			System.exit(1);
		}

		ExperimentDatabasePreparer preparer = new ExperimentDatabasePreparer(expConfig, handle);
		try {
			preparer.synchronizeExperiments();
		} catch (ExperimentDBInteractionFailedException | IllegalExperimentSetupException | AlgorithmTimeoutedException | InterruptedException | AlgorithmExecutionCanceledException
				| ExperimentAlreadyExistsInDatabaseException e) {
			LOGGER.error("Couldn't synchrinze experiment table.", e);
			System.exit(1);
		}
	}

	protected abstract void executeExperiments(final String executor, final ICoevolutionConfig expConfig, final IDatabaseConfig dbConfig, final ExperimenterMySQLHandle handle);

	protected void logFinalPipeline(final ExperimentConfiguration experimentConfiguration, final SearchResult result, final Map<String, Object> experimentDBColumns,
			final IExperimentIntermediateResultProcessor processor) throws IOException, TrainingException, PredictionException, InterruptedException, ExperimentEvaluationFailedException {
		if (result == null || result.getPipelineEvaluationReport() == null || result.getPipelineEvaluationReport().getConstructionInstruction() == null) {
			LOGGER.info("No pipeline found.");
			experimentDBColumns.put("final_pipeline", "None found.");
			experimentDBColumns.put("internal_performance", "None found.");
		} else {
			LOGGER.info("Internal performance: {} {}", result.getPipelineEvaluationReport().getPerformance(), result.getPipelineEvaluationReport().getConstructionInstruction());
			experimentDBColumns.put("final_pipeline", result.getPipelineEvaluationReport().getConstructionInstruction());
			experimentDBColumns.put("internal_performance", result.getPipelineEvaluationReport().getPerformance());
		}

		LOGGER.info("{}", experimentDBColumns);
		processor.processResults(experimentDBColumns);
		experimentDBColumns.clear();
	}

	protected void executeFinalPipeline(final ExperimentConfiguration experimentConfiguration, final SearchResult result, final Map<String, Object> experimentDBColumns,
			final IExperimentIntermediateResultProcessor processor) throws IOException, TrainingException, PredictionException, InterruptedException, ExperimentEvaluationFailedException {
		if (result == null || result.getPipelineEvaluationReport() == null || result.getPipelineEvaluationReport().getConstructionInstruction().isEmpty()
				|| result.getPipelineEvaluationReport().getImports().isEmpty()) {
			return;
		}
		ScikitLearnWrapper<IPrediction, IPredictionBatch> sklearnWrapper = new ScikitLearnWrapper<>(result.getPipelineEvaluationReport().getConstructionInstruction(),
				result.getPipelineEvaluationReport().getImports(), false, EScikitLearnProblemType.RUL);
		sklearnWrapper.setScikitLearnWrapperConfig(experimentConfiguration.getScikitLearnWrapperConfig());
		sklearnWrapper.setPythonTemplate(experimentConfiguration.getFeaturePythonTemplatePath());
		sklearnWrapper.setTimeout(experimentConfiguration.getRulTimeout());
		sklearnWrapper.setSeed(experimentConfiguration.getSeed());
		this.executeFinalPipeline(experimentConfiguration, sklearnWrapper, experimentDBColumns, processor);
	}

	protected void executeFinalPipeline(final ExperimentConfiguration experimentConfiguration, final ScikitLearnWrapper<IPrediction, IPredictionBatch> sklearnWrapper,
			final Map<String, Object> experimentDBColumns, final IExperimentIntermediateResultProcessor processor)
			throws IOException, TrainingException, InterruptedException, ExperimentEvaluationFailedException {
		if (sklearnWrapper != null) {
			try {
				IPredictionBatch predictionBatch = sklearnWrapper.fitAndPredict(experimentConfiguration.getTrainingData(), experimentConfiguration.getEvaluationData());
				List<Double> expected = experimentConfiguration.getEvaluationData().stream().map(x -> (Double) x.getLabel()).collect(Collectors.toList());
				List<IRegressionPrediction> predictions = predictionBatch.getPredictions().stream()
						.map(prediction -> new SingleTargetRegressionPrediction(Math.max(0, ((IRegressionPrediction) prediction).getDoublePrediction()))).collect(Collectors.toList());

                List<Double> predictedValues = predictions.stream().map(x -> x.getPrediction()).collect(Collectors.toList());
                File resultFile = new File(String.format("tmp/%s.csv", experimentConfiguration.getExperimentId()));
                FileUtils.writeLines(resultFile, predictedValues);

                for (ERulPerformanceMeasure performanceMeasure : ERulPerformanceMeasure.values()) {
					double performance = performanceMeasure.loss(expected, predictions);
					experimentDBColumns.put("performance_" + performanceMeasure.name().toLowerCase(), performance);
					LOGGER.info("performance_{}: {}", performanceMeasure.name().toLowerCase(), performance);
				}
			} catch (PredictionException e) {
				LOGGER.info("Evaluating final pipeline failed.", e);
				for (ERulPerformanceMeasure performanceMeasure : ERulPerformanceMeasure.values()) {
					experimentDBColumns.put("performance_" + performanceMeasure.name().toLowerCase(), "-");
				}
			}

			LOGGER.info("{}", experimentDBColumns);
			processor.processResults(experimentDBColumns);
			experimentDBColumns.clear();
		}
	}
}
