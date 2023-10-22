#include "tasksys.h"
#include <thread>
#include <mutex>
#include "CycleTimer.h"
#include <condition_variable>
#include <queue>
#include <algorithm>


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
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemSerial::sync() {
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
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
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

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): ITaskSystem(num_threads) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
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


void wait_for_lock(int& remaining_tasks, int& task_index, IRunnable*& runnable_header, IRunnable*& local_runnable, int& local_remaining_tasks, int& local_task_index, std::mutex& tasks_lock, std::condition_variable& CV, bool& destructor_called){
    std::unique_lock<std::mutex> ul(tasks_lock);
    //printf("waiting \n");
    if(destructor_called){
        return;
    }
    while(remaining_tasks <= 0){
        //printf("thread waiting for lock \n");

        if(destructor_called){
            //printf("returning from wait_for_lock because destructor called \n");
            return;
        }

        CV.wait(ul);
        
        if(destructor_called){
            //printf("returning from wait_for_lock because destructor called \n");
            return;
        }
    }
    //printf("thread got the lock \n");
    //printf("got work, %d\n", remaining_tasks);
    local_runnable = runnable_header;
    remaining_tasks--;
    task_index++;
           
    local_remaining_tasks = remaining_tasks;
    local_task_index = task_index;
}


void TaskSystemParallelThreadPoolSleeping::completition_check(){
    std::unique_lock<std::mutex> ul_completition_check(tasks_lock);
    kill_check_value++;
    //printf("kill_check_value %d remaining_tasks = %d \n", kill_check_value, remaining_tasks); 
    if(kill_check_value == num_total_tasks_header + 1){
        //printf("pushing to completed vector \n");
        completed_tasks.push_back(current_TaskID);
        //printf("completed_tasks_0_index %d \n", completed_tasks[0]);
    }
    

}

void TaskSystemParallelThreadPoolSleeping::Fetch_ready_task(){
    BulkTaskInfo rt;

    std::unique_lock<std::mutex> ul_wake(tasks_lock);
    //printf("fetch ready_task enter \n");
        if(remaining_tasks <= 0){
            //printf("Fetch ready task if statement one \n");
            if(!ready_tasks.empty()){
                //printf("Fetch ready task if statement two \n");
                rt = ready_tasks.front();
                ready_tasks.pop();
                remaining_tasks = rt.num_total_tasks + 1;
                task_index = -1;
                runnable_header = rt.runnable;
                num_total_tasks_header = rt.num_total_tasks;
                current_TaskID = rt.ID;
                kill_check_value = 0;

                if(remaining_tasks > 0){
                    //printf("notify all fetch ready task /n");
                    CV.notify_all();
                }
            }
        }
}


void TaskSystemParallelThreadPoolSleeping::grant_sync_permission(){
    std::unique_lock<std::mutex> ul_wake_sync(tasks_lock);
    if(ready_tasks.empty() && waiting_tasks.empty() && remaining_tasks <= 0 && kill_check_value == num_total_tasks_header + 1){
        //printf("granting sync permission -------------------------------------------- \n");
        CV_sync.notify_all();
    }
}

void TaskSystemParallelThreadPoolSleeping::thread_func_sleep(){
    int local_task_index = 0;
    int local_remaining_tasks = 0;
    IRunnable* local_runnable = runnable_header;

    while(true){
        if(destructor_called){
            //printf("first destructor exit \n");
            return;
        }
        //printf("before fetch_ready_task \n");
         //printf("destructor_called_before fetch_ready_task = %d \n", destructor_called);
        Fetch_ready_task();
        
  
        wait_for_lock(remaining_tasks, task_index, runnable_header, local_runnable, local_remaining_tasks, local_task_index, tasks_lock, CV, destructor_called);
        //printf("destructor_called = %d \n", destructor_called);
        if(destructor_called){
            //printf("returning from thread_func \n");
            break;
        }
        //printf("came after break \n");   
        //printf("runnable_called %p, local_task_index = %d, local_remaining_tasks = %d \n", runnable_header, local_task_index, local_remaining_tasks);
        local_runnable->runTask(local_task_index, num_total_tasks_header);
            //remaining_tasks--;
            //task_index++;
        //printf("after \n");

            
        completition_check();
        waiting_to_ready();

        grant_sync_permission();

                       // //printf("akshit_after %d", local_task_index);
        //}
        //double Time3 = CycleTimer::currentSeconds();
       // //printf(" Remaining Tasks = %d Task Index = %d \n [Lock Time and task time]: [%.3f] ms & [%.3f] ms\n", local_remaining_tasks, local_task_index, (Time2 - Time1) * 1000, (Time3 -Time2)*1000 );
    }
}



TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    //printf("constructor called \n");
    ID_counter = 0;
    max_thrds = num_threads;
    destructor_called = false;
    kill_check_value = 0;
    //ready_tasks.clear();
    while (!ready_tasks.empty()) {
        ready_tasks.pop();  
    }
    waiting_tasks.clear();
    completed_tasks.clear();
    workers.clear();
    for(int i = 0; i < max_thrds; i++){
        std::thread* new_worker = new std::thread();
        workers.push_back(new_worker);
    }
    
    //printf("constructor");
    runnable_header = nullptr;
    num_total_tasks_header = 0;
    remaining_tasks = 0;
 
    for(int i = 0; i < max_thrds-1; i++){
        *(workers[i]) = std::thread([this](){ this->thread_func_sleep(); });
        //, std::ref(runnable_header), std::ref(task_index), std::ref(tasks_lock), std::ref(num_total_tasks_header), std::ref(remaining_tasks), std::ref(destructor_called), std::ref(kill_lock), std::ref(kill_check_value), std::ref(CV), std::ref(CV_run));
       // //printf("thread %d spawaned \n", i);
    }

}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    //
    // TODO: CS149 student implementations may decide to perform cleanup
    // operations (such as thread pool shutdown construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    //printf("In the destructor \n" );
    destructor_called = true;
    CV.notify_all();
    //printf("starting to merge \n");
    for(int i = 0; i < max_thrds-1; i++){
      workers[i]->join();
    }
    //printf("Exiting from destructor");

}



void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    std::vector<TaskID> deps;
    runAsyncWithDeps(runnable, num_total_tasks, deps);
    sync();

}


void TaskSystemParallelThreadPoolSleeping::waiting_to_ready(){
    BulkTaskInfo w;// = nullptr;
    std::vector<TaskID> w_deps;
    bool completed = true;
    BulkTaskInfo rt;

    std::unique_lock<std::mutex> ul_wake(tasks_lock);
    if(remaining_tasks <= 0){
        //printf("Waiting queue size %d \n", waiting_tasks.size());
        //printf("remaining task less zero , so entering the for loop\n");
        for(int i = 0; i < waiting_tasks.size(); i++){
            //printf("entering the loop \n");
            w = waiting_tasks[i];
            w_deps = w.deps;
            completed = true;
            for(int j=0; j < w_deps.size(); j++){
                auto it = std::find(completed_tasks.begin(), completed_tasks.end(), w_deps[j]);
                if(it == completed_tasks.end()){
                    //printf("Didn't find the %d in competed task \n", w_deps[j]);
                    completed = false;
                    break;
                }
            }
            if(completed){
                waiting_tasks.erase(waiting_tasks.begin() + i);
                //printf("pushing to the ready queue \n");
                ready_tasks.push(w);
            }
        }

        //printf("after the waiting loop \n");
        if(!ready_tasks.empty()){
            rt = ready_tasks.front();
            ready_tasks.pop();
            remaining_tasks = rt.num_total_tasks + 1;
            task_index = -1;
            runnable_header = rt.runnable;
            num_total_tasks_header = rt.num_total_tasks;
            current_TaskID = rt.ID;
            kill_check_value = 0;
            if(remaining_tasks > 0){
                //printf("notifying the threads \n");
                CV.notify_all();
            }
        
        }
    }
}


TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //
    //printf("Enter run_async \n");
    BulkTaskInfo BT;// = nullptr;
    //printf("run aync waiting for the lock \n");
    tasks_lock.lock();
        BT.runnable = runnable;
        BT.num_total_tasks = num_total_tasks;
        BT.ID = ID_counter;
        ID_counter++;
        BT.deps = deps;
    
    
        //printf("Pushed a task to waiting queue \n");
        
        waiting_tasks.push_back(BT);
    tasks_lock.unlock();
    waiting_to_ready();



    // for (int i = 0; i < num_total_tasks; i++) {
    //     runnable->runTask(i, num_total_tasks);
    // }

    return BT.ID;
}

void TaskSystemParallelThreadPoolSleeping::sync() {
    std::unique_lock<std::mutex> ul_sync(tasks_lock);
    while(!ready_tasks.empty() || !waiting_tasks.empty() || remaining_tasks > 0 || (kill_check_value != num_total_tasks_header + 1)){
        //printf("waiting for sync");
        CV_sync.wait(ul_sync);
    }
    //printf("sync completed");

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //
    return;
}
