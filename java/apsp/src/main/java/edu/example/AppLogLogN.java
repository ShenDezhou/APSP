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
public class AppLogLogN {

  public static void main(String[] args) {
    log.info("epsilon=" + EarlyStopRepeatedSquareMatrixMultiplicationDistanceProduct.epsilon);
    log.info("number of rows:{}", EarlyStopRepeatedSquareMatrixMultiplicationDistanceProduct.n);
    log.info("diameter of graph:" + EarlyStopRepeatedSquareMatrixMultiplicationDistanceProduct.diameter);
    try {
      log.info(Instant.now().toString());
      INDArray adjacencyNumpy = Nd4j.createFromNpyFile(new File("src/main/resources/starsci.npy"));

      log.info("matrix loaded;");
      INDArray apas = EarlyStopRepeatedSquareMatrixMultiplicationDistanceProduct.allPairsShortestPath(adjacencyNumpy);
      log.info("distance product calculation finished;");
      Nd4j.writeAsNumpy(apas, new File("src/main/resources/result.npy"));
      log.info("write to IO finished.");
      log.info(Instant.now().toString());
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
