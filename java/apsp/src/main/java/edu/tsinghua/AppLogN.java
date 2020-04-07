package edu.tsinghua;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.time.Instant;

@Slf4j
public class ApplogN {

  public static void main(String[] args) {
    log.info("epsilon=" + EarlyStopRepeatedSquareMatrixMultiplicationDistanceProduct.epsilon);
    log.info("number of rows:{}", EarlyStopRepeatedSquareMatrixMultiplicationDistanceProduct.n);
    log.info("diameter of graph:" + EarlyStopRepeatedSquareMatrixMultiplicationDistanceProduct.diameter);
    try {
      log.info(Instant.now().toString());
      INDArray adjacencyNumpy = Nd4j.createFromNpyFile(new File("src/main/resources/starsci.npy"));

      log.info("matrix loaded;");
      INDArray apas = EarlyStopRepeatedSquareMatrixMultiplicationDistanceProduct.allPairsShortestPath(adjacencyNumpy, adjacencyNumpy.rows());
      log.info("distance product calculation finished;");
      Nd4j.writeAsNumpy(apas, new File("src/main/resources/result.npy"));
      log.info("write to IO finished.");
      log.info(Instant.now().toString());
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
