typedef struct {
  // Control work assignments
  int start, end;

  // Shared by all functions
  double *data;
  double *clusterCentroids;
  int *clusterAssignments;
  double *currCost;
  int M, N, K;
} WorkerArgs;

export void computeAssignments_ispc(uniform WorkerArgs *const args){
    double *minDist = new double[args->M];
  
    // Initialize arrays
    double startTime, endTime, Time;
    startTime = CycleTimer::currentSeconds();

    foreach (int m =0; m < args->M; m++) {
        minDist[m] = 1e30;
        args->clusterAssignments[m] = -1;
    }
    startTime = CycleTimer::currentSeconds();

    foreach (int m = 0; m < args->M; m++) {
        foreach (int k = args->start; k < args->end; k++) {  //Change 1: reordered loops to reuse m dimension instead of k. 
        double d = dist(&args->data[m * args->N],
                        &args->clusterCentroids[k * args->N], args->N);
        if (d < minDist[m]) {
            minDist[m] = d;
            args->clusterAssignments[m] = k;
        }
        }
    }

    endTime = CycleTimer::currentSeconds();
    Time = std::min(1e30, endTime - startTime);
    printf("[part2]:\t\t[%.3f] ms\n", Time * 1000);

    free(minDist);
}
