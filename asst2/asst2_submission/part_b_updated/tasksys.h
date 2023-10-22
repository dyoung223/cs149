#ifndef _TASKSYS_H
#define _TASKSYS_H

#include "itasksys.h"
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

/*
 * TaskSystemSerial: This class is the student's implementation of a
 * serial task execution engine.  See definition of ITaskSystem in
 * itasksys.h for documentation of the ITaskSystem interface.
 */
class TaskSystemSerial: public ITaskSystem {
    public:
        TaskSystemSerial(int num_threads);
        ~TaskSystemSerial();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

/*
 * TaskSystemParallelSpawn: This class is the student's implementation of a
 * parallel task execution engine that spawns threads in every run()
 * call.  See definition of ITaskSystem in itasksys.h for documentation
 * of the ITaskSystem interface.
 */
class TaskSystemParallelSpawn: public ITaskSystem {
    public:
        TaskSystemParallelSpawn(int num_threads);
        ~TaskSystemParallelSpawn();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

/*
 * TaskSystemParallelThreadPoolSpinning: This class is the student's
 * implementation of a parallel task execution engine that uses a
 * thread pool. See definition of ITaskSystem in itasksys.h for
 * documentation of the ITaskSystem interface.
 */
class TaskSystemParallelThreadPoolSpinning: public ITaskSystem {
    public:
        TaskSystemParallelThreadPoolSpinning(int num_threads);
        ~TaskSystemParallelThreadPoolSpinning();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

/*
 * TaskSystemParallelThreadPoolSleeping: This class is the student's
 * optimized implementation of a parallel task execution engine that uses
 * a thread pool. See definition of ITaskSystem in
 * itasksys.h for documentation of the ITaskSystem interface.
 */
struct BulkTaskInfo{
    IRunnable* runnable;
    int num_total_tasks;
    std::vector<TaskID> deps;
    TaskID ID;
};

class TaskSystemParallelThreadPoolSleeping: public ITaskSystem {
    public:
        TaskSystemParallelThreadPoolSleeping(int num_threads);
        ~TaskSystemParallelThreadPoolSleeping();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        void thread_func_sleep();
        void Fetch_ready_task();
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();

        void waiting_to_ready();
        void grant_sync_permission();
        void completition_check();
        int max_thrds;
        int remaining_tasks;
        std::vector <std::thread*> workers;
        IRunnable* runnable_header;
        int num_total_tasks_header;
        std::mutex tasks_lock;
        int task_index;
        bool destructor_called;
        int kill_check_value;
        std::mutex kill_lock;
        std::condition_variable CV;
        std::condition_variable CV_run;
        std::condition_variable CV_sync;
        TaskID ID_counter;
        std::queue<BulkTaskInfo> ready_tasks;
        std::vector<BulkTaskInfo> waiting_tasks;
        std::vector<TaskID> completed_tasks;
        int kill_num_total_tasks;
        TaskID current_TaskID;
};

#endif
