package edu.tsinghua;

import java.time.Duration;
import java.time.Instant;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import lombok.extern.slf4j.Slf4j;

@Slf4j
public class FloydWarshall {
  public static final float INF = Float.MAX_VALUE;
  public static INDArray allPairsShortestPath(INDArray op, INDArray op2) {
    if (op.columns() == op2.rows()) {
      INDArray op3 = Nd4j.ones(op.rows(), op2.columns()).muli(INF);
      for (int i = 0; i < op.rows(); i++) {
        for (int j = 0; j < op2.columns(); j++) {
          for (int k = 0; k < op.columns(); k++) {
            if (op3.getFloat(i, j) > op.getFloat(i, k) + op2.getFloat(k, j)) {
              op3.putScalar(new int[] { i, j }, op.getFloat(i, k) + op2.getFloat(k, j));
            }
          }
        }
      }
      log.info(op3.toString());
      return op3;
    }
    return null;
  }

  public static void main(String[] args) {
    Instant start = Instant.now();
    INDArray nd = Nd4j.create(new double[] { -1, 2, 3, 4, 5, 6 }, new int[] { 3, 2 });
    INDArray nd2 = Nd4j.create(new double[] { -1, 2, 3, 4, 5, 6 }, new int[] { 2, 3 });
    INDArray result = FloydWarshall.allPairsShortestPath(nd, nd2);
    log.info(Duration.between(start, Instant.now()).toString());
  }
}
