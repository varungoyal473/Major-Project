import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class CPUSchedulingDatasetGenerator:
    """
    Generates dataset by simulating actual CPU scheduling algorithms
    and selecting the best performer based on Composite Score (Penalty + 2*Response Time).
    """
    
    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)
        self.algorithms = ['FCFS', 'SJF', 'Priority', 'RR']
    
    def generate_processes(self, n_processes: int, scenario_type: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if scenario_type is None:
            scenario_type = np.random.randint(0, 4)
        
        # Arrivals
        if scenario_type == 0:
            arrival_times = np.sort(np.random.uniform(0, 10, n_processes))
        elif scenario_type == 1:
            arrival_times = np.sort(np.random.uniform(0, 5, n_processes))
        elif scenario_type == 2:
            arrival_times = np.sort(np.random.uniform(0, 3, n_processes))
        else:  # RR Favorable - All arrive at exactly the same time
            arrival_times = np.zeros(n_processes)
        
        # Bursts and Priorities
        if scenario_type == 2:  
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
        else:
            priorities = np.random.randint(1, 10, n_processes)
            if scenario_type == 0:
                burst_times = np.random.uniform(8, 12, n_processes)
            elif scenario_type == 1:
                short_count = n_processes // 2
                short_bursts = np.random.uniform(1, 5, short_count)
                long_bursts = np.random.uniform(15, 30, n_processes - short_count)
                burst_times = np.concatenate([short_bursts, long_bursts])
                np.random.shuffle(burst_times)
            else:  # RR Favorable - Identical bursts
                base_burst = np.random.uniform(5, 15)
                burst_times = np.full(n_processes, base_burst)
        
        burst_times = np.maximum(burst_times, 1)
        return arrival_times, burst_times, priorities
    
    def calculate_fcfs(self, arrival_times: np.ndarray, burst_times: np.ndarray, priorities: np.ndarray) -> Tuple[float, float, float, float, float]:
        n = len(arrival_times)
        waiting_times = np.zeros(n)
        penalties = np.zeros(n)
        response_times = np.zeros(n)
        completion_time = 0
        
        for i in range(n):
            if completion_time < arrival_times[i]:
                completion_time = arrival_times[i]
            waiting_times[i] = completion_time - arrival_times[i]
            penalties[i] = waiting_times[i] * (11 - priorities[i])
            response_times[i] = waiting_times[i]
            completion_time += burst_times[i]
            
        turnaround_times = waiting_times + burst_times
        composite_score = np.mean(penalties) + (np.mean(response_times) * 2)
        return np.mean(waiting_times), np.mean(turnaround_times), np.mean(penalties), np.mean(response_times), composite_score
    
    def calculate_sjf(self, arrival_times: np.ndarray, burst_times: np.ndarray, priorities: np.ndarray) -> Tuple[float, float, float, float, float]:
        n = len(arrival_times)
        waiting_times = np.zeros(n)
        penalties = np.zeros(n)
        response_times = np.zeros(n)
        completed = np.zeros(n, dtype=bool)
        current_time = 0
        completed_count = 0
        
        while completed_count < n:
            available = [i for i in range(n) if not completed[i] and arrival_times[i] <= current_time]
            
            if not available:
                for i in range(n):
                    if not completed[i]:
                        current_time = arrival_times[i]
                        break
                continue
            
            min_burst_idx = min(available, key=lambda x: burst_times[x])
            waiting_times[min_burst_idx] = current_time - arrival_times[min_burst_idx]
            penalties[min_burst_idx] = waiting_times[min_burst_idx] * (11 - priorities[min_burst_idx])
            response_times[min_burst_idx] = waiting_times[min_burst_idx]
            current_time += burst_times[min_burst_idx]
            completed[min_burst_idx] = True
            completed_count += 1
            
        turnaround_times = waiting_times + burst_times
        composite_score = np.mean(penalties) + (np.mean(response_times) * 2)
        return np.mean(waiting_times), np.mean(turnaround_times), np.mean(penalties), np.mean(response_times), composite_score
    
    def calculate_priority(self, arrival_times: np.ndarray, burst_times: np.ndarray, priorities: np.ndarray) -> Tuple[float, float, float, float, float]:
        n = len(arrival_times)
        waiting_times = np.zeros(n)
        penalties = np.zeros(n)
        response_times = np.zeros(n)
        completed = np.zeros(n, dtype=bool)
        current_time = 0
        completed_count = 0
        
        while completed_count < n:
            available = [i for i in range(n) if not completed[i] and arrival_times[i] <= current_time]
            
            if not available:
                for i in range(n):
                    if not completed[i]:
                        current_time = arrival_times[i]
                        break
                continue
            
            high_priority_idx = min(available, key=lambda x: priorities[x])
            waiting_times[high_priority_idx] = current_time - arrival_times[high_priority_idx]
            penalties[high_priority_idx] = waiting_times[high_priority_idx] * (11 - priorities[high_priority_idx])
            response_times[high_priority_idx] = waiting_times[high_priority_idx]
            current_time += burst_times[high_priority_idx]
            completed[high_priority_idx] = True
            completed_count += 1
            
        turnaround_times = waiting_times + burst_times
        composite_score = np.mean(penalties) + (np.mean(response_times) * 2)
        return np.mean(waiting_times), np.mean(turnaround_times), np.mean(penalties), np.mean(response_times), composite_score
    
    def calculate_rr(self, arrival_times: np.ndarray, burst_times: np.ndarray, priorities: np.ndarray, time_quantum: float) -> Tuple[float, float, float, float, float]:
        n = len(arrival_times)
        remaining_burst = burst_times.copy()
        waiting_times = np.zeros(n)
        penalties = np.zeros(n)
        response_times = np.full(n, -1.0)
        last_execution = np.zeros(n)
        queue = []
        current_time = 0
        completed = 0
        
        for i in range(n):
            if arrival_times[i] <= current_time:
                queue.append(i)
        
        while completed < n:
            if not queue:
                for i in range(n):
                    if remaining_burst[i] > 0 and arrival_times[i] > current_time:
                        current_time = arrival_times[i]
                        queue.append(i)
                        break
                continue
            
            process_id = queue.pop(0)
            
            # Record response time on first touch
            if response_times[process_id] == -1.0:
                response_times[process_id] = current_time - arrival_times[process_id]
            
            if last_execution[process_id] > 0:
                waiting_times[process_id] += current_time - last_execution[process_id]
            else:
                waiting_times[process_id] = current_time - arrival_times[process_id]
            
            execution_time = min(time_quantum, remaining_burst[process_id])
            current_time += execution_time
            remaining_burst[process_id] -= execution_time
            
            for i in range(n):
                if (remaining_burst[i] > 0 and i not in queue and i != process_id and arrival_times[i] <= current_time):
                    queue.append(i)
            
            last_execution[process_id] = current_time
            
            if remaining_burst[process_id] > 0:
                queue.append(process_id)
            else:
                completed += 1
                
        for i in range(n):
            penalties[i] = waiting_times[i] * (11 - priorities[i])
            
        turnaround_times = waiting_times + burst_times
        composite_score = np.mean(penalties) + (np.mean(response_times) * 2)
        return np.mean(waiting_times), np.mean(turnaround_times), np.mean(penalties), np.mean(response_times), composite_score
    
    def generate_scenario_features(self, arrival_times: np.ndarray, burst_times: np.ndarray, 
                                  priorities: np.ndarray, time_quantum: float) -> Dict[str, float]:
        return {
            'num_processes': len(arrival_times),
            'avg_burst_time': np.mean(burst_times),
            'std_burst_time': np.std(burst_times),
            'min_burst_time': np.min(burst_times),
            'max_burst_time': np.max(burst_times),
            'avg_arrival_time': np.mean(arrival_times),
            'arrival_spread': np.max(arrival_times) - np.min(arrival_times),
            'avg_inter_arrival': np.mean(np.diff(arrival_times)) if len(arrival_times) > 1 else 0,
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
    
    def generate_dataset(self, n_samples: int = 1000, force_balanced: bool = True) -> pd.DataFrame:
        data_rows = []
        algorithm_counts = {algo: 0 for algo in self.algorithms}
        target_per_algo = n_samples // 4
        
        print(f"Generating {n_samples} samples...")
        sample_count = 0
        attempts = 0
        max_attempts = n_samples * 10
        
        while sample_count < n_samples and attempts < max_attempts:
            attempts += 1
            if force_balanced:
                min_algo_count = min(algorithm_counts.values())
                needed_algos = [algo for algo, count in algorithm_counts.items() if count == min_algo_count]
                if 'Priority' in needed_algos and algorithm_counts['Priority'] < target_per_algo:
                    scenario_type = 2
                elif 'SJF' in needed_algos and algorithm_counts['SJF'] < target_per_algo:
                    scenario_type = 1
                elif 'FCFS' in needed_algos and algorithm_counts['FCFS'] < target_per_algo:
                    scenario_type = 0
                elif 'RR' in needed_algos and algorithm_counts['RR'] < target_per_algo:
                    scenario_type = 3
                else:
                    scenario_type = np.random.randint(0, 4)
            else:
                scenario_type = np.random.randint(0, 4)
            
            n_processes = np.random.randint(3, 15)
            arrival_times, burst_times, priorities = self.generate_processes(n_processes, scenario_type)
            time_quantum = np.mean(burst_times) * np.random.uniform(0.8, 1.2) if scenario_type == 3 else np.random.uniform(2, 10)
            
            # Fetch the Composite Score (Index 4)
            fcfs_metrics = self.calculate_fcfs(arrival_times, burst_times, priorities)
            sjf_metrics = self.calculate_sjf(arrival_times, burst_times, priorities)
            prio_metrics = self.calculate_priority(arrival_times, burst_times, priorities)
            rr_metrics = self.calculate_rr(arrival_times, burst_times, priorities, time_quantum)
            
            composites = {
                'FCFS': fcfs_metrics[4],
                'SJF': sjf_metrics[4],
                'Priority': prio_metrics[4],
                'RR': rr_metrics[4]
            }
            
            best_algorithm = min(composites, key=composites.get)
            
            if force_balanced and algorithm_counts[best_algorithm] >= target_per_algo and min(algorithm_counts.values()) < target_per_algo:
                continue
            
            features = self.generate_scenario_features(arrival_times, burst_times, priorities, time_quantum)
            
            # Save composite metrics for visualization
            features['fcfs_composite'] = fcfs_metrics[4]
            features['sjf_composite'] = sjf_metrics[4]
            features['priority_composite'] = prio_metrics[4]
            features['rr_composite'] = rr_metrics[4]
            features['best_algorithm'] = best_algorithm
            
            data_rows.append(features)
            algorithm_counts[best_algorithm] += 1
            sample_count += 1
            
            if sample_count >= n_samples: break
            if sample_count % 100 == 0: print(f"  Processed {sample_count}/{n_samples} samples...")
        
        df = pd.DataFrame(data_rows)
        # Fast Balancing logic
        if force_balanced and len(df) > 0:
            target_count = int(df['best_algorithm'].value_counts().mean())
            balanced_dfs = []
            for algo in self.algorithms:
                algo_df = df[df['best_algorithm'] == algo]
                if len(algo_df) < target_count and len(algo_df) > 0:
                    needed = target_count - len(algo_df)
                    oversampled = algo_df.sample(n=min(needed, len(algo_df)), replace=True).copy()
                    num_cols = [c for c in oversampled.select_dtypes(include=[np.number]).columns if not c.endswith('_composite')]
                    for c in num_cols: oversampled[c] *= np.random.uniform(0.98, 1.02, len(oversampled))
                    algo_df = pd.concat([algo_df, oversampled])
                elif len(algo_df) > target_count:
                    algo_df = algo_df.sample(n=target_count)
                balanced_dfs.append(algo_df)
            df = pd.concat(balanced_dfs, ignore_index=True)
            
        print(f"Dataset generation complete! Generated {len(df)} samples.")
        return df.sample(frac=1).reset_index(drop=True)
    
    def print_summary(self, df: pd.DataFrame):
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print("\nAlgorithm Distribution (Based on Composite Score):")
        for algo in self.algorithms:
            count = (df['best_algorithm'] == algo).sum()
            print(f"  {algo:8s}: {count:4d} samples ({(count / len(df)) * 100:.1f}%)")
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = 'cpu_scheduling_dataset.csv'):
        # Crucial: Drop the composite answers so ML model doesn't cheat
        save_df = df.drop(['fcfs_composite', 'sjf_composite', 'priority_composite', 'rr_composite'], axis=1, errors='ignore')
        save_df.to_csv(filename, index=False)
        print(f"\nDataset saved to '{filename}'")
        return save_df

def create_cpu_scheduling_dataset(n_samples: int = 1000, save_csv: bool = False):
    generator = CPUSchedulingDatasetGenerator(random_seed=42)
    df = generator.generate_dataset(n_samples, force_balanced=True)
    generator.print_summary(df)
    if save_csv: return generator.save_to_csv(df)
    return df.drop(['fcfs_composite', 'sjf_composite', 'priority_composite', 'rr_composite'], axis=1, errors='ignore')

if __name__ == "__main__":
    dataset = create_cpu_scheduling_dataset(n_samples=2500, save_csv=True)