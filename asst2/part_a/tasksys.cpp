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
    
    std::mutex my_lock;
    int current_iteration_task = 0;
    std::thread threads_handle[my_threads_available];
    for (int j = 0; j < my_threads_available; j++){
        threads_handle[j] = std::thread(dynamic_allocation_run, std::ref(runnable), num_total_tasks, std::ref(current_iteration_task), std::ref(my_lock));
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

    
    IRunnable* local_runnable = global_runnable; 
    while(true){
        if(kill_threads) break;
        my_lock.lock();
        if(global_current_task_iteration < global_num_total_tasks){
            local_runnable = global_runnable;
            local_task_index = global_current_task_iteration;
            global_current_task_iteration++;
            local_total_tasks = global_num_total_tasks;
        }
        
        my_lock.unlock();
         
        if(local_task_index < local_total_tasks){
            if(local_runnable != nullptr){
                
                local_runnable->runTask(local_task_index, local_total_tasks);

                kill_lock.lock();
                kill_check_value++;
                kill_lock.unlock();
                
                local_task_index = local_total_tasks + 1;
            }
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
    IRunnable* global_runnable = nullptr;
    global_num_total_tasks = 0;
    global_current_task_iteration = 0;
    kill_threads = false;
    kill_check_value = -1;

    for(int j = 0; j < num_threads; j++){
        thread_handle.push_back(std::thread([this](){ 
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
    
    global_runnable = std::ref(runnable);
    global_current_task_iteration = 0;
    global_num_total_tasks = num_total_tasks;
    kill_check_value = 0;
    
    my_lock.unlock();

    while(true){
        kill_lock.lock();
        if(kill_check_value == num_total_tasks){
            kill_lock.unlock();
            break;
        }
        kill_lock.unlock();
    }
  
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
void TaskSystemParallelThreadPoolSleeping::dynamic_allocation_w_sleep(){
    int local_task_index = 0; 
    int local_total_tasks = 0;

    
    IRunnable* local_runnable = global_runnable; 
    while(true){
        if(kill_threads) break;
        my_lock.lock();
        if(global_current_task_iteration < global_num_total_tasks){
            local_runnable = global_runnable;
            local_task_index = global_current_task_iteration;
            global_current_task_iteration++;
            local_total_tasks = global_num_total_tasks;
        }
        
        my_lock.unlock();
         
        if(local_task_index < local_total_tasks){
            if(local_runnable != nullptr){
                
                local_runnable->runTask(local_task_index, local_total_tasks);

                local_task_index = local_total_tasks + 1;

                kill_count_lock.lock();
                kill_check_value++;
                
                if(kill_check_value == local_total_tasks){
                    kill_count_lock.unlock();
                    wait_for_complete.lock();
                    wait_for_complete.unlock();
                    complete.notify_all();
                }else{

                    kill_count_lock.unlock();
                }


            }
        }else{
            //no work to do, go to sleep
            std::unique_lock<std::mutex> lk(sleep_lock);
            sleep.wait(lk);
            lk.unlock();
        }
    }
}
TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //

    my_threads_available = num_threads;
    IRunnable* global_runnable = nullptr;
    global_num_total_tasks = 0;
    global_current_task_iteration = 0;
    kill_threads = false;
    kill_check_value = -1;

    for(int j = 0; j < num_threads; j++){
        thread_handle.push_back(std::thread([this](){ 
            this->dynamic_allocation_w_sleep();
           }));
    }


}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    //
    // TODO: CS149 student implementations may decide to perform cleanup
    // operations (such as thread pool shutdown construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    //printf("Destructor starting\n");
    kill_threads = true;
    for(int i = 0; i < my_threads_available; i++){
        sleep.notify_all();
    }
    for(int i = 0; i < my_threads_available; i++){
        thread_handle.at(i).join();
    }
    //printf("Destructor finished\n");
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    //printf("start run function\n");
    std::unique_lock<std::mutex> lock(wait_for_complete);
    my_lock.lock();
    
    global_runnable = std::ref(runnable);
    global_current_task_iteration = 0;
    global_num_total_tasks = num_total_tasks;
    kill_check_value = 0;
    
    my_lock.unlock();

    for(int i = 0; i < my_threads_available; i++){
        sleep.notify_all();
    }
    //go to sleep until threads notify they are all done
    complete.wait(lock);
    lock.unlock();
    //printf("end run function\n");
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
