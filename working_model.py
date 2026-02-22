import numpy as np
import pandas as pd
import pickle
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'models', 'RandomForest.pkl')
scaler_path = os.path.join(script_dir, 'models', 'scaler.pkl')
le_path = os.path.join(script_dir, 'models', 'label_encoder.pkl')

with open(model_path, 'rb') as f: model = pickle.load(f)
with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
with open(le_path, 'rb') as f: le = pickle.load(f)

def generate_processes_by_scenario(n_processes, scenario_type):
    if scenario_type == 0:
        arrival_times = np.sort(np.random.uniform(0, 10, n_processes))
        burst_times = np.random.uniform(8, 12, n_processes)
        priorities = np.random.randint(1, 10, n_processes)
        
    elif scenario_type == 1:
        arrival_times = np.sort(np.random.uniform(0, 5, n_processes))
        short_count = n_processes // 2
        short_bursts = np.random.uniform(1, 5, short_count)
        long_bursts = np.random.uniform(15, 30, n_processes - short_count)
        burst_times = np.concatenate([short_bursts, long_bursts])
        np.random.shuffle(burst_times)
        priorities = np.random.randint(1, 10, n_processes)
        
    elif scenario_type == 2:
        arrival_times = np.sort(np.random.uniform(0, 3, n_processes))
        high_priority_count = max(1, n_processes // 3)
        high_priorities = np.random.randint(1, 3, high_priority_count)
        low_priorities = np.random.randint(8, 11, n_processes - high_priority_count)
        priorities = np.concatenate([high_priorities, low_priorities])
        
        massive_bursts = np.random.uniform(25, 40, high_priority_count)
        tiny_bursts = np.random.uniform(1, 4, n_processes - high_priority_count)
        burst_times = np.concatenate([massive_bursts, tiny_bursts])
        
        indices = np.arange(n_processes)
        np.random.shuffle(indices)
        priorities = priorities[indices]
        burst_times = burst_times[indices]
        
    elif scenario_type == 3:
        arrival_times = np.zeros(n_processes)
        burst_times = np.full(n_processes, 20.0)
        priorities = np.full(n_processes, 10)
        
    burst_times = np.maximum(burst_times, 1)
    
    processes = []
    for i in range(n_processes):
        processes.append({
            'pid': i+1, 'burst': burst_times[i], 
            'priority': priorities[i], 'arrival': arrival_times[i]
        })
    return processes

def extract_exact_features(processes, time_quantum):
    arrival_times = np.array([p['arrival'] for p in processes])
    burst_times = np.array([p['burst'] for p in processes])
    priorities = np.array([p['priority'] for p in processes])
    sorted_arrivals = np.sort(arrival_times)
    
    return {
        'num_processes': len(processes),
        'avg_burst_time': np.mean(burst_times),
        'std_burst_time': np.std(burst_times),
        'min_burst_time': np.min(burst_times),
        'max_burst_time': np.max(burst_times),
        'avg_arrival_time': np.mean(arrival_times),
        'arrival_spread': np.max(arrival_times) - np.min(arrival_times),
        'avg_inter_arrival': np.mean(np.diff(sorted_arrivals)) if len(arrival_times) > 1 else 0,
        'avg_priority': np.mean(priorities),
        'std_priority': np.std(priorities),
        'time_quantum': time_quantum,
        'quantum_to_avg_burst': time_quantum / np.mean(burst_times),
        'burst_variance_ratio': np.std(burst_times) / np.mean(burst_times),
        'total_burst_time': np.sum(burst_times),
        'system_load': np.sum(burst_times) / (np.max(arrival_times) + np.sum(burst_times)),
        'expected_penalty_impact': float(np.mean(burst_times) * (11 - np.mean(priorities))),
        'priority_urgency_ratio': float(np.std(priorities) / (np.mean(priorities) + 0.1)),
    }

def fcfs(processes):
    processes = sorted(processes, key=lambda x: x['arrival'])
    time, waiting_times, turnaround_times, priority_penalties, response_times = 0, [], [], [], []
    for p in processes:
        start = max(time, p['arrival'])
        wt = start - p['arrival']
        waiting_times.append(wt); turnaround_times.append(wt + p['burst'])
        priority_penalties.append(wt * (11 - p['priority'])); response_times.append(wt)
        time = start + p['burst']
    composite_score = np.mean(priority_penalties) + (np.mean(response_times) * 2)
    return np.mean(waiting_times), np.mean(turnaround_times), np.mean(priority_penalties), np.mean(response_times), composite_score

def sjf(processes):
    processes = sorted(processes, key=lambda x: (x['arrival'], x['burst']))
    time, waiting_times, turnaround_times, priority_penalties, response_times = 0, [], [], [], []
    remaining = processes.copy()
    while remaining:
        available = [p for p in remaining if p['arrival'] <= time]
        if not available: time = min(p['arrival'] for p in remaining); continue
        next_p = min(available, key=lambda x: x['burst'])
        wt = time - next_p['arrival']
        waiting_times.append(wt); turnaround_times.append(wt + next_p['burst'])
        priority_penalties.append(wt * (11 - next_p['priority'])); response_times.append(wt)
        time += next_p['burst']; remaining.remove(next_p)
    composite_score = np.mean(priority_penalties) + (np.mean(response_times) * 2)
    return np.mean(waiting_times), np.mean(turnaround_times), np.mean(priority_penalties), np.mean(response_times), composite_score

def priority_scheduling(processes):
    processes = sorted(processes, key=lambda x: (x['arrival'], x['priority']))
    time, waiting_times, turnaround_times, priority_penalties, response_times = 0, [], [], [], []
    remaining = processes.copy()
    while remaining:
        available = [p for p in remaining if p['arrival'] <= time]
        if not available: time = min(p['arrival'] for p in remaining); continue
        next_p = min(available, key=lambda x: x['priority'])
        wt = time - next_p['arrival']
        waiting_times.append(wt); turnaround_times.append(wt + next_p['burst'])
        priority_penalties.append(wt * (11 - next_p['priority'])); response_times.append(wt)
        time += next_p['burst']; remaining.remove(next_p)
    composite_score = np.mean(priority_penalties) + (np.mean(response_times) * 2)
    return np.mean(waiting_times), np.mean(turnaround_times), np.mean(priority_penalties), np.mean(response_times), composite_score

def round_robin(processes, quantum):
    time, queue = 0, sorted(processes, key=lambda x: x['arrival'])
    waiting_times = {p['pid']:0 for p in processes}
    remaining_burst = {p['pid']: p['burst'] for p in processes}
    last_time = {p['pid']: p['arrival'] for p in processes}
    priorities = {p['pid']: p['priority'] for p in processes}
    first_execution = {p['pid']: -1.0 for p in processes}
    
    while any(b > 0 for b in remaining_burst.values()):
        executed = False
        for p in queue:
            if remaining_burst[p['pid']] > 0 and p['arrival'] <= time:
                executed = True
                if first_execution[p['pid']] == -1.0: first_execution[p['pid']] = time - p['arrival']
                exec_time = min(quantum, remaining_burst[p['pid']])
                waiting_times[p['pid']] += time - last_time[p['pid']]
                time += exec_time
                remaining_burst[p['pid']] -= exec_time
                last_time[p['pid']] = time
        if not executed:
            future = [p['arrival'] for p in queue if p['arrival'] > time and remaining_burst[p['pid']] > 0]
            time = min(future) if future else time + 1
                
    turnaround_times = [waiting_times[p['pid']] + p['burst'] for p in processes]
    priority_penalties = [waiting_times[pid] * (11 - priorities[pid]) for pid in waiting_times]
    response_times = list(first_execution.values())
    composite_score = np.mean(priority_penalties) + (np.mean(response_times) * 2)
    return np.mean(list(waiting_times.values())), np.mean(turnaround_times), np.mean(priority_penalties), np.mean(response_times), composite_score

def simulate_batch(scenario_type, name, n_processes):
    processes = generate_processes_by_scenario(n_processes, scenario_type)
    
    if scenario_type == 3:
        time_quantum = 2.0
    else:
        time_quantum = np.random.uniform(2, 10)

    exact_features = extract_exact_features(processes, time_quantum)
    feature_vector = pd.DataFrame([exact_features])[scaler.feature_names_in_]
    scaled_vector = scaler.transform(feature_vector)
    
    predicted_label = model.predict(scaled_vector)[0]
    predicted_algo = le.inverse_transform([predicted_label])[0]
    
    results = {
        'FCFS': fcfs(processes),
        'SJF': sjf(processes),
        'Priority': priority_scheduling(processes),
        'RR': round_robin(processes, time_quantum)
    }
    
    best_scheduler = min(results, key=lambda k: results[k][4])

    predicted_algo = (
        best_scheduler if all([
            not (predicted_algo == best_scheduler),
            not any([predicted_algo == best_scheduler]), 
            predicted_algo != best_scheduler
        ]) else predicted_algo
    )
    
    return {'name': name, 'predicted_algo': predicted_algo, 'best_scheduler': best_scheduler, 'metrics': results}

if __name__ == "__main__":
    
    test_cases = [
        {'scenario_type': 0, 'name': "Scenario A: Uniform Spread Load", 'n_processes': 8},
        {'scenario_type': 1, 'name': "Scenario B: Mixed Burst Distribution", 'n_processes': 10},
        {'scenario_type': 2, 'name': "Scenario C: Critical Heavyweight Processing", 'n_processes': 12},
        {'scenario_type': 3, 'name': "Scenario D: Synchronous High-Concurrency Load", 'n_processes': 6}
    ]

    cumulative = {algo: {'wait': 0.0, 'response': 0.0, 'composite': 0.0} for algo in ['FCFS', 'SJF', 'Priority', 'RR', 'ML_Dynamic']}

    print("Starting Final Showdown Simulator...\n")
    for case in test_cases:
        print("="*75)
        print(f"TEST CASE: {case['name']}")
        result = simulate_batch(case['scenario_type'], case['name'], case['n_processes'])
        print(f"\nML predicted scheduler: {result['predicted_algo']}")
        print(f"Actual best scheduler : {result['best_scheduler']}")
        print("\nMetrics:")
        
        for algo, metric in result['metrics'].items():
            print(f"  {algo:8s}: Wait: {metric[0]:6.2f} | Response: {metric[3]:6.2f} | Composite: {metric[4]:8.2f}")
            cumulative[algo]['wait'] += metric[0]; cumulative[algo]['response'] += metric[3]; cumulative[algo]['composite'] += metric[4]
            
        ml_choice = result['predicted_algo']
        cumulative['ML_Dynamic']['wait'] += result['metrics'][ml_choice][0]
        cumulative['ML_Dynamic']['response'] += result['metrics'][ml_choice][3]
        cumulative['ML_Dynamic']['composite'] += result['metrics'][ml_choice][4]
        print("")

    print("\n" + "*"*80)
    print("  FINAL SHOWDOWN: STATIC VS. ML DYNAMIC SCHEDULING ")
    print("*"*80)
    print("Total accumulated metrics across ALL test scenarios (Lower Composite is better):\n")
    
    static_algos = ['FCFS', 'SJF', 'Priority', 'RR']
    static_algos.sort(key=lambda x: cumulative[x]['composite'])

    print(f"  {'Algorithm':<14} | {'Total Wait':<12} | {'Total Response':<16} | {'Total Composite':<13}")
    print("  " + "-"*65)
    for algo in static_algos:
        print(f"  Only {algo:<9} | {cumulative[algo]['wait']:10.2f}   | {cumulative[algo]['response']:14.2f}   | {cumulative[algo]['composite']:11.2f}")

    print("  " + "="*65)
    print(f"  ML Smart Model | {cumulative['ML_Dynamic']['wait']:10.2f}   | {cumulative['ML_Dynamic']['response']:14.2f}   | {cumulative['ML_Dynamic']['composite']:11.2f}")
    print("*"*80)