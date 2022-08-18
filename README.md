用ortools和灰狼算法求解JSP和FJSP问题。

#if data_inital[2] == '1':
在程序开始时，将输入数据的第一行第三元素，做一个判断，如果工序的平均机器数量为1，则进入 MinimalJobshopSat()函数，用ortools求解JSP问题。
ortools作为完备算法，求解JSP问题非常强力，针对FT和LA级别的数据集，均能在短时间内求出最优解，这是常规的元启发式算法难以企及的。

如果是FJSP问题，则用灰狼算法解决问题，本次选择灰狼算法的原因是，查询近年来核心期刊文献，灰狼算法在求解调度问题中，逐渐占据优势，多数的FJSP问题的发刊文章，
都是来自灰狼算法。



灰狼算法的实现逻辑
1）社会等级分层
GWO的优化过程主要有每代种群中的最好三匹狼（具体构建时表示为三个最好的解）来指导完成。
2）包围猎物
灰狼捜索猎物时会逐渐地接近猎物并包围它，该行为的数学模型如下：
![image](https://user-images.githubusercontent.com/71360947/185347334-34fad858-a57d-4e6d-9aec-e24565851e91.png)
3）狩猎行为的数学模型在这里插入图片描述
![image](https://user-images.githubusercontent.com/71360947/185347438-be5620e7-ddfa-438a-83aa-a58599957b08.png)

放上链接：https://blog.csdn.net/welcome_yu/article/details/112095902


#代码及注释
from __future__ import print_function
import time
import chardet
import datetime
import collections
# Import Python wrapper for or-tools CP-SAT solver.
from ortools.sat.python import cp_model
import numpy as np
import random
import sys
import warnings
warnings.filterwarnings("ignore")
dt = datetime.datetime
time_delta = datetime.timedelta

#根据函数输入最优结果，进行排产操作。
def final_caculate(job, machine, machine_time, machine_num, job_num):
    jobtime = np.zeros((1, job_num))
    tmm = np.zeros((1, machine_num))
    tmmw = np.zeros((1, machine_num))
    job=list(job[0])
    for i in range(len(job)):
        job[i]=int(job[i])

    startime = 0
    list_Machine, list_Starttime, list_W = [], [], []
    for i in range(len(job)):
        svg, sig = int(job[i]), int(machine[0, i]) - 1
        if (jobtime[0, svg] > 0):  # 第svg个机器的启动运行时间
            startime = max(jobtime[0, svg], tmm[0, sig])
            tmm[0, sig] = startime + machine_time[0, i]
            jobtime[0, svg] = startime + machine_time[0, i]
        if (jobtime[0, svg] == 0):
            startime = tmm[0, sig]
            tmm[0, sig] = startime + machine_time[0, i]
            jobtime[0, svg] = startime + machine_time[0, i]

        tmmw[0, sig] += machine_time[0, i]
        list_Machine.append(machine[0, i])
        list_Starttime.append(startime)
        list_W.append(machine_time[0, i])

    job_frequency = [0] * len(job)

    for i in range(job_num):
        t = 1
        for j in range(len(job)):
            a = job[j]
            if a == i:
                job_frequency[j] = t
                t += 1

    for i in range(len(job)):
        job[i]=str(job[i]+1).rjust(3, '0')
        job_frequency[i]=str(job_frequency[i]).rjust(2, '0')

    df_result = []
    df_temp = []
    for j in range(len(job)):
        process_tuple = (
            f'S{job[j]}',
            f'S{job[j]}'+f'{job_frequency[j]}',
            int(list_Starttime[j]),
            int(list_Starttime[j] + list_W[j])
        )
        df_temp.append(process_tuple)
    f = open(sys.argv[2], "w")
    for i in range(machine_num):
        list_c = []
        for j in range(len(df_temp)):
            if i + 1 == list_Machine[j]:
                list_c.append(df_temp[j])
        strs=''
        for k in range(len(list_c)):
            if k==0:
                strs=f'{list_c[k]}'
            else:
                strs = strs + ',' + f'{list_c[k]}'
        f.write(f'M{i+1}'+':' +strs+'\n')
    f.close()

#灰狼算法类
class GWO_algorithm():
    def __init__(self, job_num, machine_num, p, iter, popsize_num):
        self.job_num = job_num  # 工件数
        self.machine_num = machine_num  # 机器数
        self.pi = p  #交叉概率
        self.generation = iter  # 迭代次数
        self.popsize = popsize_num  # 种群规模

    #分开输出每个工件的可用机器，可用机器时间，工序可用机器编号
    def coding(self, tr1):
        sigdex, mac, mact, sdx = [], [], [], []
        sigal = tr1[0]
        tr1 = tr1[1:len(tr1) + 1]
        index = 0
        for j in range(sigal):
            sig = tr1[index]
            sdx.append(sig)
            sigdex.append(index)
            index = index + 1 + 2 * sig
        for ij in range(sigal):
            del tr1[sigdex[ij] - ij]
        for ii in range(0, len(tr1) - 1, 2):
            mac.append(tr1[ii])
            mact.append(tr1[ii + 1])
        return mac, mact, sdx
    #输出原始机器矩阵，机器时间矩阵，机器选择矩阵，放入load_data_GWO中
    def tcaculate(self, strt):
        widthx = []
        for i in range(self.job_num):
            mac, mact, sdx = self.coding(strt[i])
            siga = len(mac)
            widthx.append(siga)
        width = max(widthx)

        Tmachine, Tmachinetime = np.zeros((self.job_num, width)), np.zeros((self.job_num, width))
        tdx = []
        for i in range(self.job_num):
            mac, mact, sdx = self.coding(strt[i])
            tdx.append(sdx)
            siga = len(mac)
            Tmachine[i, 0:siga] = mac
            Tmachinetime[i, 0:siga] = mact
        return Tmachine, Tmachinetime, tdx
    #从input.txt中读取数据，输出原始机器矩阵，机器时间矩阵，机器选择矩阵，独立工作选择和工作时常。
    def load_data_GWO(self):
        f = open(sys.argv[1])
        f1 = f.readlines()
        c, count = [], 0
        for line in f1:
            t1 = line.strip('\n')
            if (count > 0):
                a = list(map(int, t1.split()))
                c.append(a)
            count += 1
        strt = c
        Tmachine, Tmachinetime, tdx = self.tcaculate(strt)
        to, tom, work = 0, [], []
        for i in range(self.job_num):
            to += len(tdx[i])
            tim = []
            for j in range(1, len(tdx[i]) + 1, 1):
                tim.append(sum(tdx[i][0:j]))
                work.append(i)
            tom.append(tim)
        Tmachine=Tmachine+1
        return Tmachine, Tmachinetime, tdx, work, tom

    # 返回当前迭代种群机器编码和机器时间编码的列表形式。
    def MA_MAtime_List(self, W1, M1, T1):  # 把加工机器编码和加工时间编码转化为对应列表，目的是记录工件的加工时间和加工机器
        Ma_W1, Tm_W1, WCross = [], [], []
        for i in range(self.job_num):  # 添加工件个数的空列表
            Ma_W1.append([]), Tm_W1.append([]), WCross.append([])
        for i in range(W1.shape[1]):
            signal1 = int(W1[0, i]) - 1
            Ma_W1[signal1].append(M1[0, i]), Tm_W1[signal1].append(T1[0, i])  # 记录每个工件的加工机器
            index = np.random.randint(0, 2, 1)[0]
            WCross[signal1].append(index)  # 随机生成一个为0或者1的列表，用于后续的机器的均匀交叉
        return Ma_W1, Tm_W1, WCross

    #返回单个例子机器编码和机器时间编码的列表形式。
    def MA_MAtime(self, W1, Ma_W1, Tm_W1):  # 列表返回机器及加工时间编码
        memory1 = np.zeros((1, self.job_num), dtype=np.int)
        m1, t1 = np.zeros((1, W1.shape[1])), np.zeros((1, W1.shape[1]))
        for i in range(W1.shape[1]):
            signal1 = int(W1[0, i]) - 1
            m1[0, i] = Ma_W1[signal1][memory1[0, signal1]]  # 读取对应工序的加工机器
            t1[0, i] = Tm_W1[signal1][memory1[0, signal1]]
            memory1[0, signal1] += 1
        return m1, t1

    #用于交换工序编码，产生种群多样性。
    def machine_cross(self, Ma_W1, Tm_W1, Ma_W2, Tm_W2, WCross):  # 机器均匀交叉
        MC1, MC2, TC1, TC2 = [], [], [], []
        for i in range(self.job_num):
            MC1.append([]), MC2.append([]), TC1.append([]), TC2.append([])
            for j in range(len(WCross[i])):
                if (WCross[i][j] == 0):  # 为0时继承另一个父代的加工机器选择
                    MC1[i].append(Ma_W1[i][j]), MC2[i].append(Ma_W2[i][j]), TC1[i].append(Tm_W1[i][j]), TC2[i].append(
                        Tm_W2[i][j])
                else:  # 为1时继承父代的机器选择
                    MC2[i].append(Ma_W1[i][j]), MC1[i].append(Ma_W2[i][j]), TC2[i].append(Tm_W1[i][j]), TC1[i].append(
                        Tm_W2[i][j])
        return MC1, TC1, MC2, TC2

    #用于创造工序编码，机器编码，时间编码，利于后续灰狼算法的运算。
    def creat_process(self):
        initial_a = np.random.rand(len(self.work))
        index_work = np.array(initial_a).argsort()
        job = []
        for i in range(len(self.work)):
            job.append(self.work[index_work[i]])
        job = np.array(job).reshape(1, len(self.work))

        ccount = np.zeros((1, self.job_num), dtype=np.int)
        machine = np.ones((1, job.shape[1]))
        machine_time = np.ones((1, job.shape[1]))  # 初始化矩阵
        for i in range(job.shape[1]):
            oper = int(job[0, i])
            highs = self.tom[oper][ccount[0, oper]]
            lows = self.tom[oper][ccount[0, oper]] - self.tdx[oper][ccount[0, oper]]
            n_machine = self.Tmachine[oper, lows:highs]
            n_time = self.Tmachinetime[oper, lows:highs]
            ccount[0, oper] += 1
            if np.random.rand() > self.pi:  # 选取最小加工时间机器
                machine_time[0, i] = min(n_time)
                index = np.argwhere(n_time == machine_time[0, i])
                machine[0, i] = n_machine[index[0, 0]]
            else:  # 否则随机挑选机器
                index = np.random.randint(0, len(n_time), 1)
                machine[0, i] = n_machine[index[0]]
                machine_time[0, i] = n_time[index[0]]
        return job, machine, machine_time, initial_a
    #用于评估当前排列工序的最优解。
    def caculate(self, job, machine, machine_time):
        jobtime = np.zeros((1, self.job_num))
        tmm = np.zeros((1, self.machine_num))
        tmmw = np.zeros((1, self.machine_num))
        startime = 0
        list_M, list_S, list_W = [], [], []
        for i in range(job.shape[1]):
            svg, sig = int(job[0, i]), int(machine[0, i]) - 1
            if (jobtime[0, svg] > 0):
                startime = max(jobtime[0, svg], tmm[0, sig])
                tmm[0, sig] = startime + machine_time[0, i]
                jobtime[0, svg] = startime + machine_time[0, i]
            if (jobtime[0, svg] == 0):
                startime = tmm[0, sig]
                tmm[0, sig] = startime + machine_time[0, i]
                jobtime[0, svg] = startime + machine_time[0, i]

            tmmw[0, sig] += machine_time[0, i]
            list_M.append(machine[0, i])
            list_S.append(startime)
            list_W.append(machine_time[0, i])

        tmax = np.argmax(tmm[0]) + 1  # 结束最晚的机器
        C_finish = max(tmm[0])  # 最晚完工时间
        return C_finish, list_M, list_S, list_W, tmax
    #运行函数，记录灰狼的迭代次数和输出最终最优结果。
    def gwo_result(self):
        Tmachine, Tmachinetime, tdx, work, tom = self.load_data_GWO()
        parm_data = [Tmachine, Tmachinetime, tdx, work, tom]
        self.Tmachine, self.Tmachinetime, self.tdx, self.work, self.tom = parm_data[0], parm_data[1], parm_data[2], \
                                                                          parm_data[3], parm_data[4]
        answer, result = [], []
        job_init = np.zeros((self.popsize, len(work)))
        work_job, work_M, work_T = np.zeros((self.popsize, len(work))), np.zeros((self.popsize, len(work))), np.zeros(
            (self.popsize, len(work)))
        for gen in range(self.generation):
            if (gen < 1):  # 第一次生成多个可行的工序编码，机器编码，时间编码
                for i in range(self.popsize):
                    job, machine, machine_time, initial_a = self.creat_process()
                    C_finish, _, _, _, _ = self.caculate(job, machine, machine_time)
                    answer.append(C_finish)
                    work_job[i], work_M[i], work_T[i] = job[0], machine[0], machine_time[0]
                    job_init[i] = initial_a
                print('种群初始的最优解:%.0f' % (min(answer)))
                result.append([gen, min(answer)])  # 记录初始解的最小完工时间

            index_sort = np.array(answer).argsort()  # 返回完工时间从小到大的位置索引
            work_job1, work_M1, work_T1 = work_job[index_sort], work_M[index_sort], work_T[index_sort]
            answer1 = np.array(answer)[index_sort]
            job_init1 = job_init[index_sort]

            Alpha = job_init1[0]  # α狼
            Beta = job_init1[1]  # β狼
            Delta = job_init1[2]  # δ狼
            a = 2 * (1 - gen / self.generation)

            for i in range(3, self.popsize):  # 用最优位置进行工序编码的更新
                job, machine, machine_time = work_job1[i:i + 1], work_M1[i:i + 1], work_T1[i:i + 1]
                Ma_W1, Tm_W1, WCross = self.MA_MAtime_List(job, machine, machine_time)
                x = job_init1[i]

                r1 = random.random()  # 灰狼算法解的更新
                r2 = random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = C1 * Alpha - x
                x1 = x - A1 * D_alpha

                r1 = random.random()
                r2 = random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = C2 * Beta - x
                x2 = x - A2 * D_beta

                r1 = random.random()
                r2 = random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_alpha = C3 * Delta - x
                x3 = x - A3 * D_alpha

                initial_a = (x1 + x2 + x3) / 3  # 更新公式
                index_work = initial_a.argsort()
                job_new = []
                for j in range(len(work)):
                    job_new.append(work[index_work[j]])
                job_new = np.array(job_new).reshape(1, len(work))
                machine_new, time_new = self.MA_MAtime(job_new, Ma_W1, Tm_W1)
                C_finish, _, _, _, _ = self.caculate(job_new, machine_new, time_new)

                work_job1[i] = job_new[0]  # 更新工序编码
                job_init1[i] = initial_a
                work_M1[i], work_T1[i] = machine_new[0], time_new[0]
                answer1[i] = C_finish
            for i in range(0, self.popsize, 2):
                job, machine, machine_time = work_job1[i:i + 1], work_M1[i:i + 1], work_T1[i:i + 1]
                Ma_W1, Tm_W1, WCross = self.MA_MAtime_List(job, machine, machine_time)
                job1, machine1, machine_time1 = work_job1[i + 1:i + 2], work_M1[i + 1:i + 2], work_T1[i + 1:i + 2]
                Ma_W2, Tm_W2, WCross = self.MA_MAtime_List(job1, machine1, machine_time1)

                MC1, TC1, MC2, TC2 = self.machine_cross(Ma_W1, Tm_W1, Ma_W2, Tm_W2, WCross)
                machine_new, time_new = self.MA_MAtime(job, MC1, TC1)
                C_finish, _, _, _, _ = self.caculate(job, machine_new, time_new)
                if (C_finish < answer1[i]):  # 如果更新后的完工时间大于原解，更新机器和加工时间编码
                    work_M1[i] = machine_new[0]
                    work_T1[i] = time_new[0]
                    answer1[i] = C_finish
                machine_new1, time_new1 = self.MA_MAtime(job1, MC2, TC2)
                C_finish, _, _, _, _ = self.caculate(job1, machine_new1, time_new1)
                if (C_finish < answer1[i + 1]):  # 如果更新后的完工时间大于原解，更新机器和加工时间编码
                    work_M1[i + 1] = machine_new1[0]
                    work_T1[i + 1] = time_new1[0]
                    answer1[i + 1] = C_finish
            work_job, work_M, work_T = work_job1, work_M1, work_T1
            answer = answer1
            job_init = job_init1
            result.append([gen + 1, min(answer)])  # 记录每一次迭代的最优个体
            print('灰狼算法第%.0f次迭代的最优解:%.0f' % (gen + 1, min(answer)))
            best = answer.tolist().index(min(answer))

        workpiece_number, machine_number, machine_time=np.array([work_job[best]]), np.array([work_M[best]]), np.array([work_T[best]])
        result=np.array(result).reshape(len(result), 2)[iter,1]

        return workpiece_number, machine_number, machine_time, result

#用于初始读取数据，判断是JSP还是FJSP问题。
def load_text(file_name):
    try:
        with open(sys.argv[1], "rb") as f:
            f_read = f.read()
            f_cha_info = chardet.detect(f_read)
            final_data = f_read.decode(f_cha_info['encoding'])
            return final_data, True
    except FileNotFoundError:
        return str(None), False

#ORTOOLS运行函数
def MinimalJobshopSat(string):
    a = []
    for i in string:
        a.append(int(i))
    job_num, machine_number = a[0], a[1]
    for _ in range(3):
        a.pop(0)
    for i in range(job_num):
        a.pop(3 * job_num * i)
    all_machines = range(machine_number)
    jobs_data = []
    job = []
    for i, (j, k) in enumerate(zip(a[1::3], a[2::3])):
        job.append((j - 1, k))
        if (i + 1) % machine_number == 0:
            jobs_data.append(job)
            job = []

    """Minimal jobshop problem."""
    # Create the model.
    model = cp_model.CpModel()

    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(task[1] for job in jobs_data for task in job)

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval')

    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = '_%i_%i' % (job_id, task_id)
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var)
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id +
                                1].start >= all_tasks[job_id, task_id].end)

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')

    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(jobs_data)
    ])

    model.Minimize(obj_var)

    # Solve model.
    solver = cp_model.CpSolver()

    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    assigned_task_type(
                        start=solver.Value(all_tasks[job_id, task_id].start),
                        job=job_id,
                        index=task_id,
                        duration=task[1]))

        # Create per machine output lines.
        print('User time: %.2fs' % solver.UserTime())
        print('Wall time: %.2fs' % solver.WallTime())
        print('Optimal Schedule Length: %i' % solver.ObjectiveValue())
        #进行排产
        df_result = []
        for machine in all_machines[::1]:
            df_result_temp = []
            assigned_jobs[machine].sort()
            for assigned_task in assigned_jobs[machine]:
                df_result_tuple = (
                    assigned_task.job + 1, assigned_task.index + 1,
                    assigned_task.start, assigned_task.start + assigned_task.duration
                )
                df_result_temp.append(df_result_tuple)
            df_result_dic = {f'M{machine + 1}': df_result_temp}
            df_result.append(df_result_dic)
        #输出到输出文件
        f = open(sys.argv[2], "w")
        for line in df_result:
            f.write(str(line) + '\n')
        f.close()


if __name__ == '__main__':

    time_begin = time.time()
    file_name = sys.argv[1]  #输入文件
    data_input, check = load_text(file_name)
    data_inital = list(map(str, data_input.split()))
    job_num, machine_num = int(data_inital[0]), int(data_inital[1]) #job_num 工件个数 machine_num 机器个数

    #判断是否为FJSP问题
    if data_inital[2] == '1':
        #ortools运行
        MinimalJobshopSat(data_inital)
    else:
        #GWO算法运行
        p = 0.5  # 灰狼算法选择概率
        iter = 400  # 灰狼迭代次数
        num = 50  # 灰狼种群数量
        #创建灰狼算法类
        ho = GWO_algorithm(job_num, machine_num, p, iter, num)
        #输出最终工序码，机器码和机器时间，以及最终结果。
        workpiece_number, machine_number, machine_time, result = ho.gwo_result()
        #进行排产
        final_caculate(workpiece_number, machine_number, machine_time, machine_num, job_num)
        time_end = time.time()
        all_time = time_end - time_begin
        #输出结果和时间
        f = open(sys.argv[2], "a")
        f.write('最优解：'+str(int(result))+ '\n')
        f.write('所用时间：'+'{:.0f}分 {:.0f}秒'.format( all_time  // 60, all_time  % 60)+ '\n')
        f.close()




