#include "tasksys.h"
#include <thread>
#include <mutex>
IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) {}
ITaskSystem::~ITaskSystem() {}

/*
 * ================================================================
 * Serial task system implementation
 * ================================================================
 */

const char* TaskSystemSerial::name() {
    return "Serial";
}

TaskSystemSerial::TaskSystemSerial(int num_threads): ITaskSystem(num_threads) {
}

TaskSystemSerial::~TaskSystemSerial() {}

void TaskSystemSerial::run(IRunnable* runnable, int num_total_tasks) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemSerial::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                          const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemSerial::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelSpawn::name() {
    return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    my_threads_available = num_threads;

}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}


void run_set_of_tasks(IRunnable* runnable, int num_total_tasks, int start_index, int run_count){
    for(int i = start_index; i < start_index + run_count; i++){
        runnable->runTask(i, num_total_tasks);
    }
}
void fine_granularity_run(IRunnable* runnable, int num_total_tasks, int threadID, int num_threads_avail){
    for(int i = threadID; i < num_total_tasks; i+=num_threads_avail){
        runnable->runTask(i, num_total_tasks);
    }
}
void dynamic_allocation_run(IRunnable* runnable, int num_total_tasks, int& current_iteration_task, std::mutex& my_lock){
    int local_task_index; 
    while(true){
        my_lock.lock();
        if(current_iteration_task < num_total_tasks){
            local_task_index = current_iteration_task;
            current_iteration_task++;
        }else{
            my_lock.unlock();
            return;
        }
        my_lock.unlock();
        runnable->runTask(local_task_index, num_total_tasks);
    }
}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    
/*
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
    */
    /*
    std::thread threads_handle[num_total_tasks];
    //printf("num_total_tasks %i\n", num_total_tasks);
    for (int i = 0; i < num_total_tasks; i++){
        threads_handle[i] = std::thread(&IRunnable::runTask, std::ref(runnable),  i, num_total_tasks);
    }
    for (int i = 0; i < num_total_tasks; i++){
        threads_handle[i].join();
    }
    */
    std::mutex my_lock;
    int current_iteration_task = 0;
    std::thread threads_handle[my_threads_available];
    for (int j = 0; j < my_threads_available; j++){
        threads_handle[j] = std::thread(dynamic_allocation_run, std::ref(runnable), num_total_tasks, std::ref(current_iteration_task), std::ref(my_lock));
        //threads_handle[j] = std::thread(fine_granularity_run, std::ref(runnable), num_total_tasks, j, my_threads_available);
        //threads_handle[j] = std::thread(run_set_of_tasks, std::ref(runnable), num_total_tasks, int(j*num_total_tasks/my_threads_available), int(num_total_tasks/my_threads_available));
        //printf("starting threads %i, threads to run %i", int(j*num_total_tasks/my_threads_available), int(num_total_tasks/my_threads_available));
    }

    for(int j = 0; j < my_threads_available; j++){
        threads_handle[j].join();
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Spinning Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSpinning::name() {
    return "Parallel + Thread Pool + Spin";
}
void TaskSystemParallelThreadPoolSpinning::dynamic_allocation_run_for_spin(){
    int local_task_index = 0; 
    int local_total_tasks = 0;

    
    IRunnable* local_runnable = fake_runnable; 
    //printf("Current iterationt ask pointer: %x\n", std::ref(current_task_iteration));
    while(true){
        //my_lock.lock();
        if(kill_threads) break;
        my_lock.lock();
        //printf("Current iteration task: %i, num total tasks %i\n", current_task_iteration, num_total_tasks);
        if(current_task_iteration < fake_num_total_tasks){
            local_runnable = fake_runnable;
            local_task_index = current_task_iteration;
            current_task_iteration++;
            local_total_tasks = fake_num_total_tasks;
            //printf("local task index %i, num total task %i\n", local_task_index, fake_num_total_tasks);
            //printf("Current iteration task: %i, num total tasks %i\n", current_task_iteration, num_total_tasks);
        }
        
        my_lock.unlock();
         
        if(local_task_index < local_total_tasks){
            //my_lock.unlock();
            if(local_runnable != nullptr){
                
                //printf(" runnable is not a nullptr\n");
                //printf("running: local task index %i, num total task %i\n", local_task_index, fake_num_total_tasks);
                //printf("VALID Pointer: %x\n", std::ref(my_runnable));
                //my_lock.unlock();
                local_runnable->runTask(local_task_index, local_total_tasks);
                //my_lock.unlock();
                kill_lock.lock();
                kill_check_value++;
                kill_lock.unlock();
                
                local_task_index = local_total_tasks + 1;
            }else{
                //my_lock.unlock();
                //printf("runnable is a nullptr\n");
                //printf("value of nullptr %x \n", fake_runnable);
            }
        
        }else{
            //my_lock.unlock();
        }
        
    }
}
TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    my_threads_available = num_threads;
    //num_total_tasks_left = 0;
    IRunnable* fake_runnable = nullptr;
    fake_num_total_tasks = 0;
    current_task_iteration = 0;
    kill_threads = false;
    kill_check_value = -1;

    //printf("Current iteration task: %x, num total tasks %i\n", std::ref(current_task_iteration), fake_num_total_tasks);
    //printf("fake runnable %x\n", std::ref(fake_runnable));
    for(int j = 0; j < num_threads; j++){
        thread_handle.push_back(std::thread([this](){ //TaskSystemParallelThreadPoolSpinning::dynamic_allocation_run_for_spin));
            this->dynamic_allocation_run_for_spin();
           }));
    }

}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {
    kill_threads = true;
    for(int i = 0; i < my_threads_available; i++){
        thread_handle.at(i).join();
    }
}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    my_lock.lock();
    
    fake_runnable = std::ref(runnable);
    //printf("fake runnable in run: %x\n", std::ref(fake_runnable));
    //printf("current task iter pointer %x\n", std::ref(current_task_iteration));
    current_task_iteration = 0;
    fake_num_total_tasks = num_total_tasks;
    kill_check_value = 0;
    my_lock.unlock();
    //printf("Setting runnable pointer from away from nullptr to: %x\n", std::ref(runnable));
    while(true){
        kill_lock.lock();
        if(kill_check_value == num_total_tasks){
            kill_lock.unlock();
            //printf("Current task completed \n");
            break;
        }/*else{
            sleep(10);
        }*/
        kill_lock.unlock();
    }
    //for (int i = 0; i < num_total_tasks; i++) {
    //    runnable->runTask(i, num_total_tasks);
    //}
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSleeping::name() {
    return "Parallel + Thread Pool + Sleep";
}

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    //
    // TODO: CS149 student implementations may decide to perform cleanup
    // operations (such as thread pool shutdown construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //

    return 0;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //

    return;
}
