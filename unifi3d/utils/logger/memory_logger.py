import torch


class MemoryLogger:
    def __init__(self, name):
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_allocated = torch.cuda.memory_allocated(self.device) / (1024**2)
        self.init_reserved = torch.cuda.memory_reserved(self.device) / (1024**2)
        self.logs = {}

    def log_memory(self, description):
        all_allocated = torch.cuda.memory_allocated(self.device) / (1024**2)
        all_reserved = torch.cuda.memory_reserved(self.device) / (1024**2)
        allocated_diff = all_allocated - self.init_allocated
        reserved_diff = all_reserved - self.init_reserved
        self.logs[description] = {
            "allocated_diff": allocated_diff,
            "reserved_diff": reserved_diff,
            "total_allocated": all_allocated,
            "total_reserved": all_reserved,
        }
        # Update initial memory usage for next log
        self.init_allocated = all_allocated
        self.init_reserved = all_reserved

    def print_gpu_memory(self):
        all_allocated = torch.cuda.memory_allocated(self.device)
        all_reserved = torch.cuda.memory_reserved(self.device)
        print(f"  Total Allocated GPU memory: {all_allocated  / (1024 ** 3):.2f} GB")
        print(f"  Total Reserved GPU memory: {all_reserved  / (1024 ** 3):.2f} GB")

    def print_logs(self):
        total_allocated = 0
        total_reserved = 0
        for description, log in self.logs.items():
            print(f"{description}:")
            print(f"  Allocated total: {log['allocated_diff'] :.2f} MB")
            print(f"  Reserved total: {log['reserved_diff'] :.2f} MB")
            total_allocated += log["allocated_diff"]
            total_reserved += log["reserved_diff"]

        print(f"Total Memory Used:")
        print(f"  Total Allocated: {total_allocated :.2f} MB")
        print(f"  Total Reserved: {total_reserved :.2f} MB")
