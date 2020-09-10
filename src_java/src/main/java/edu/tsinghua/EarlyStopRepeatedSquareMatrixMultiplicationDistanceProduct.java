//Copyright (c) 2020-2040 Dezhou Shen, Tsinghua University
//        Licensed under the Apache License, Version 2.0 (the "License");
//        you may not use this file except in compliance with the License.
//        You may obtain a copy of the License at
//        http://www.apache.org/licenses/LICENSE-2.0
//        Unless required by applicable law or agreed to in writing, software
//        distributed under the License is distributed on an "AS IS" BASIS,
//        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//        See the License for the specific language governing permissions and
//        limitations under the License.
package edu.tsinghua;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

//Server CPU:13:38:33.241 ~ 13:39:18.465 = 45s
//Server GPU:13:41:15.716 ~ 13:42:50.337 = 95s
@Slf4j
public class EarlyStopRepeatedSquareMatrixMultiplicationDistanceProduct {
  public static final float INF = Float.MAX_VALUE;
  public static final float epsilon = 0.0000001f;
  public static final float n = 8380;
  public static final double diameter = 8;

  public static double max(INDArray op) {
    INDArray index_op = CustomOperations.finds(op,
            aDouble -> aDouble < INF);

    double min = index_op.minNumber().doubleValue();
    double max = index_op.maxNumber().doubleValue();
    log.info("Min element:" + min + ",Max element:" + max);
    return max;
  }


  public static INDArray exponent(INDArray op, double base, double max) {
    for (int i = 0; i < op.rows(); i++) {
      for (int j = 0; j < op.columns(); j++) {
        if (Math.abs(op.getDouble(i, j)) < INF) {
          op.putScalar(new int[] { i, j }, Math.pow(base + 1, max - op.getDouble(i, j)));
        } else {
          op.putScalar(new int[] { i, j }, 0);
        }
      }
    }

//    INDArray index_op = CustomOperations.find(op,
//            (Predicate<Double>) aDouble -> aDouble < INF);
//    INDArray rindex_op = CustomOperations.find(op,
//            (Predicate<Double>) aDouble -> aDouble >= INF);
//
//    index_op = Nd4j.ones(index_op.shape()).mul(max).sub(index_op);
//    index_op = Transforms.pow(Nd4j.ones(index_op.shape()).mul(base), index_op);
//    rindex_op = Nd4j.zeros(rindex_op.shape());
    log.info("exp:"+op.toString());
    return op;
  }

  public static INDArray logarithm(INDArray op, double base, double max) {
    for (int i = 0; i < op.rows(); i++) {
      for (int j = 0; j < op.columns(); j++) {
        double d = op.getDouble(i, j);
        if (d > epsilon && !Double.isInfinite(d)) {
          op.putScalar(new int[] { i, j }, 2 * max - Math.floor(Math.log(d) / Math.log(base + 1)));

        } else {
          op.putScalar(new int[] { i, j }, INF);
        }
      }
    }
//    INDArray index_op = CustomOperations.find(op,
//            (Predicate<Double>) aDouble -> aDouble >= 0d);
//    INDArray rindex_op = CustomOperations.find(op,
//            (Predicate<Double>) aDouble -> Math.abs(aDouble) < epsilon);
//
//    index_op = Transforms.log(index_op, base);
//    index_op =  Nd4j.ones(index_op.shape()).mul(2*max).sub(index_op);
//    rindex_op = Nd4j.ones(rindex_op.shape()).mul(INF);
    log.info("log:"+op.toString());
    return op;
  }

  public static INDArray distanceProduct(INDArray op) {
    int m = op.rows();
    double op_max = max(op);
    op = exponent(op, m, op_max);
    op = Nd4j.matmul(op,op);
    op = logarithm(op, m, op_max);
    log.info("dp:"+op.toString());
    return op;
  }

  public static INDArray distanceProduct(INDArray op, double diameter) {
    int m = op.rows();
    double op_max = diameter;
    op = exponent(op, m, op_max);
    op = Nd4j.matmul(op,op);
    op = logarithm(op, m, op_max);
    log.info("dp:"+op.toString());
    return op;
  }

  @Override
  public String toString() {
    return "epsilon:" + epsilon + ";" + " rows:" + n + ";" + " diameter:" + diameter + ".";
  }

  /**
   * Use Network/Graph Diameter instead of Node Number to get faster speed for APSP.
   * @param op
   * @return
   */
  public static INDArray allPairsShortestPath(INDArray op) {
    log.info("Loop N:" + Math.ceil(Math.log(diameter) / Math.log(2)));
    for (int l = 0; l < Math.ceil(Math.log(diameter) / Math.log(2)); l++) {
      INDArray dp = distanceProduct(op.dup(), diameter);
      dp = Transforms.min(op, dp);
      op = dp;
    }
    log.info("apsp:"+op.toString());
    return op;
  }

  /**
   * User Early stop to get exact APSP result.
   * @param op
   * @param N_nodes
   * @return
   */
  public static INDArray allPairsShortestPath(INDArray op, int N_nodes) {
    log.info("Loop N:" + Math.ceil(Math.log(N_nodes) / Math.log(2)));
    for (int l = 0; l < Math.ceil(Math.log(N_nodes) / Math.log(2)); l++) {
      INDArray dp = distanceProduct(op.dup(), diameter);
      dp = Transforms.min(op, dp);
      if(dp.eps(op).all()){
        log.info("early stopped.");
        op = dp;
        break;
      }
      op = dp;
    }
    log.info("apsp:"+op.toString());
    return op;
  }
}
