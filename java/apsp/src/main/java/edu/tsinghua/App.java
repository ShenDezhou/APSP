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

import java.io.File;
import java.io.IOException;
import java.time.Instant;

@Slf4j
public class App {

  public static void main(String[] args) {

    String adjacent_numpy_matrix = args[0];
    int diameter = Integer.parseInt(args[1]);
    String apsp_numpy_matrix = args[2];
    String algorithm = "";
    if (args.length>3)
      algorithm = args[3];

    log.info("epsilon=" + EarlyStopRepeatedSquareMatrixMultiplicationDistanceProduct.epsilon);
    log.info("diameter of graph:{}", diameter);
    log.info("algorithm:{}", algorithm);
    try {
      log.info(Instant.now().toString());
      INDArray adjacencyNumpy = Nd4j.createFromNpyFile(new File(adjacent_numpy_matrix));
      log.info("number of rows:{}", adjacencyNumpy.shape()[0]);
      log.info("matrix loaded;");
      long startTime = System.currentTimeMillis();
      INDArray apsp=null;
      if (algorithm.equals("floydwarshall")) {
        log.info("algorithm:{}", algorithm);
        apsp = FloydWarshall.allPairsShortestPath(adjacencyNumpy, adjacencyNumpy);
      } else {
        apsp = EarlyStopRepeatedSquareMatrixMultiplicationDistanceProduct.allPairsShortestPath(adjacencyNumpy, diameter);
      }
      log.info("Total execution time: " + (System.currentTimeMillis()-startTime) + "ms");
      log.info("distance product calculation finished;");
      Nd4j.writeAsNumpy(apsp, new File(apsp_numpy_matrix));
      log.info("write to IO finished.");
      log.info(Instant.now().toString());
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
