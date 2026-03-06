"""
Experiment Manager for Heart Disease Diagnosis Project
Quản lý và tracking tất cả experiments với logging system hoàn chỉnh
"""

import os
import json
import numpy as np
import pandas as pd
import random
import logging
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ExperimentManager:
    """
    Quản lý và tracking tất cả experiments
    - Set global seed cho reproducibility
    - Log experiment config, params, metrics
    - Compare experiments và tạo reports
    """
    
    def __init__(self, experiment_dir="experiments"):
        """
        Initialize Experiment Manager
        
        Args:
            experiment_dir (str): Thư mục lưu experiments
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Tạo các folder con
        self.logs_dir = self.experiment_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        self.results_dir = self.experiment_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.reports_dir = self.experiment_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # File lưu experiment log
        self.experiment_log_file = self.experiment_dir / "experiment_log.json"
        
        # Setup logging system
        self._setup_logging()
        
        # Load existing experiments nếu có
        self.experiments = self._load_experiments()
        
    def _setup_logging(self):
        """Setup logging system"""
        log_file = self.logs_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Experiment Manager initialized at {self.experiment_dir}")
        
    def _load_experiments(self):
        """Load existing experiments từ file"""
        if self.experiment_log_file.exists():
            try:
                with open(self.experiment_log_file, 'r') as f:
                    experiments = json.load(f)
                self.logger.info(f"Loaded {len(experiments)} existing experiments")
                return experiments
            except Exception as e:
                self.logger.error(f"Error loading experiments: {str(e)}")
                return []
        return []
    
    def _save_experiments(self):
        """Save experiments to file"""
        try:
            with open(self.experiment_log_file, 'w') as f:
                json.dump(self.experiments, f, indent=2)
            self.logger.info(f"Saved {len(self.experiments)} experiments to {self.experiment_log_file}")
        except Exception as e:
            self.logger.error(f"Error saving experiments: {str(e)}")
    
    def set_global_seed(self, seed=42):
        """
        Đảm bảo reproducibility cho tất cả random operations
        
        Args:
            seed (int): Seed value
        """
        self.logger.info(f"Setting global seed to {seed}")
        
        # Python random
        random.seed(seed)
        
        # Numpy
        np.random.seed(seed)
        
        # Scikit-learn (thông qua numpy)
        # Đã được set thông qua np.random.seed
        
        # TensorFlow/Keras (nếu có)
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
            self.logger.info("TensorFlow seed set")
        except ImportError:
            pass
        
        # PyTorch (nếu có)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            self.logger.info("PyTorch seed set")
        except ImportError:
            pass
        
        # Environment variable cho hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        self.logger.info("Global seed set successfully")
        
    def log_experiment(self, 
                      model_name, 
                      dataset_type, 
                      params, 
                      metrics,
                      training_time=None,
                      additional_info=None):
        """
        Log một experiment với đầy đủ thông tin
        
        Args:
            model_name (str): Tên model (lr, rf, xgb, ada, gb, dt, knn, nb, svm, ensemble)
            dataset_type (str): Loại dataset (raw, fe, dt, fe_dt, master_boruta)
            params (dict): Hyperparameters của model
            metrics (dict): Metrics (accuracy, precision, recall, f1, auc)
            training_time (float): Thời gian training (giây)
            additional_info (dict): Thông tin bổ sung
        """
        experiment = {
            "experiment_id": len(self.experiments) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "dataset_type": dataset_type,
            "hyperparameters": params,
            "metrics": metrics,
            "training_time": training_time
        }
        
        if additional_info:
            experiment["additional_info"] = additional_info
        
        self.experiments.append(experiment)
        self._save_experiments()
        
        self.logger.info(f"Logged experiment {experiment['experiment_id']}: "
                        f"{model_name} on {dataset_type} - "
                        f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        
        return experiment["experiment_id"]
    
    def compare_experiments(self, 
                           filter_model=None, 
                           filter_dataset=None,
                           sort_by='accuracy',
                           ascending=False):
        """
        So sánh các experiments
        
        Args:
            filter_model (str): Lọc theo model name
            filter_dataset (str): Lọc theo dataset type
            sort_by (str): Metric để sort ('accuracy', 'f1', 'auc', 'training_time')
            ascending (bool): Sort tăng dần hay giảm dần
            
        Returns:
            pd.DataFrame: Bảng so sánh experiments
        """
        if not self.experiments:
            self.logger.warning("No experiments to compare")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.experiments)
        
        # Filter
        if filter_model:
            df = df[df['model_name'] == filter_model]
        if filter_dataset:
            df = df[df['dataset_type'] == filter_dataset]
        
        # Extract metrics to separate columns
        if 'metrics' in df.columns:
            metrics_df = pd.json_normalize(df['metrics'])
            df = pd.concat([df.drop('metrics', axis=1), metrics_df], axis=1)
        
        # Sort
        if sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=ascending)
        
        self.logger.info(f"Compared {len(df)} experiments")
        
        return df
    
    def get_best_experiment(self, metric='accuracy', filter_model=None, filter_dataset=None):
        """
        Lấy experiment tốt nhất theo metric
        
        Args:
            metric (str): Metric để đánh giá
            filter_model (str): Lọc theo model
            filter_dataset (str): Lọc theo dataset
            
        Returns:
            dict: Best experiment
        """
        df = self.compare_experiments(filter_model, filter_dataset, sort_by=metric, ascending=False)
        
        if df.empty:
            return None
        
        best = df.iloc[0].to_dict()
        self.logger.info(f"Best experiment: {best.get('model_name')} on {best.get('dataset_type')} "
                        f"- {metric}: {best.get(metric):.4f}")
        
        return best
    
    def get_summary_statistics(self):
        """
        Tính toán summary statistics cho tất cả experiments
        
        Returns:
            dict: Summary statistics
        """
        if not self.experiments:
            return {}
        
        df = self.compare_experiments()
        
        summary = {
            "total_experiments": len(df),
            "models_tested": df['model_name'].unique().tolist() if 'model_name' in df.columns else [],
            "datasets_tested": df['dataset_type'].unique().tolist() if 'dataset_type' in df.columns else [],
            "metrics_summary": {}
        }
        
        # Tính statistics cho các metrics
        metric_columns = ['accuracy', 'f1', 'auc', 'precision', 'recall', 'training_time']
        for col in metric_columns:
            if col in df.columns:
                summary['metrics_summary'][col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max())
                }
        
        return summary
    
    def generate_report(self, output_format='html'):
        """
        Tạo báo cáo HTML/Markdown tổng hợp
        
        Args:
            output_format (str): 'html' hoặc 'markdown'
            
        Returns:
            str: Path to report file
        """
        if not self.experiments:
            self.logger.warning("No experiments to generate report")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get comparison dataframe
        df = self.compare_experiments()
        
        # Get summary statistics
        summary = self.get_summary_statistics()
        
        if output_format == 'html':
            report_file = self.reports_dir / f"experiment_report_{timestamp}.html"
            self._generate_html_report(df, summary, report_file)
        else:
            report_file = self.reports_dir / f"experiment_report_{timestamp}.md"
            self._generate_markdown_report(df, summary, report_file)
        
        self.logger.info(f"Report generated: {report_file}")
        return str(report_file)
    
    def _generate_html_report(self, df, summary, output_file):
        """Generate HTML report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Heart Disease Diagnosis - Experiment Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1976d2;
            text-align: center;
            border-bottom: 3px solid #1976d2;
            padding-bottom: 15px;
        }}
        h2 {{
            color: #424242;
            margin-top: 30px;
            border-left: 5px solid #1976d2;
            padding-left: 15px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .summary-card .value {{
            font-size: 28px;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }}
        th {{
            background-color: #1976d2;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric {{
            font-weight: bold;
            color: #1976d2;
        }}
        .timestamp {{
            color: #666;
            font-size: 12px;
            text-align: center;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🫀 Heart Disease Diagnosis - Experiment Report</h1>
        <p style="text-align: center; color: #666;">
            Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
            Team: Dũng, Anh, Vinh, Hằng, Huy | AIO2025 VietAI
        </p>
        
        <h2>📊 Summary Statistics</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Total Experiments</h3>
                <div class="value">{summary.get('total_experiments', 0)}</div>
            </div>
            <div class="summary-card">
                <h3>Models Tested</h3>
                <div class="value">{len(summary.get('models_tested', []))}</div>
            </div>
            <div class="summary-card">
                <h3>Datasets Tested</h3>
                <div class="value">{len(summary.get('datasets_tested', []))}</div>
            </div>
        </div>
        
        <h2>🏆 Best Performing Models</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Model</th>
                <th>Dataset</th>
                <th>Accuracy</th>
                <th>F1 Score</th>
                <th>AUC</th>
                <th>Training Time (s)</th>
            </tr>
"""
        
        # Add top 10 experiments
        top_experiments = df.head(10)
        for idx, row in top_experiments.iterrows():
            html_content += f"""
            <tr>
                <td>{idx + 1}</td>
                <td><strong>{row.get('model_name', 'N/A')}</strong></td>
                <td>{row.get('dataset_type', 'N/A')}</td>
                <td class="metric">{row.get('accuracy', 0):.4f}</td>
                <td class="metric">{row.get('f1', 0):.4f}</td>
                <td class="metric">{row.get('auc', 0):.4f}</td>
                <td>{row.get('training_time', 0):.2f}</td>
            </tr>
"""
        
        html_content += """
        </table>
        
        <h2>📈 Performance by Model</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Avg Accuracy</th>
                <th>Avg F1</th>
                <th>Best Dataset</th>
            </tr>
"""
        
        # Group by model
        if 'model_name' in df.columns and 'accuracy' in df.columns:
            model_stats = df.groupby('model_name').agg({
                'accuracy': 'mean',
                'f1': 'mean'
            }).round(4)
            
            for model, stats in model_stats.iterrows():
                best_dataset = df[df['model_name'] == model].nlargest(1, 'accuracy')['dataset_type'].values[0] if len(df[df['model_name'] == model]) > 0 else 'N/A'
                html_content += f"""
            <tr>
                <td><strong>{model}</strong></td>
                <td class="metric">{stats['accuracy']:.4f}</td>
                <td class="metric">{stats['f1']:.4f}</td>
                <td>{best_dataset}</td>
            </tr>
"""
        
        html_content += """
        </table>
        
        <div class="timestamp">
            Report generated by Experiment Manager<br>
            Heart Disease Diagnosis Project - AIO2025
        </div>
    </div>
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_markdown_report(self, df, summary, output_file):
        """Generate Markdown report"""
        md_content = f"""# 🫀 Heart Disease Diagnosis - Experiment Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Team:** Dũng, Anh, Vinh, Hằng, Huy | AIO2025 VietAI

---

## 📊 Summary Statistics

- **Total Experiments:** {summary.get('total_experiments', 0)}
- **Models Tested:** {len(summary.get('models_tested', []))} ({', '.join(summary.get('models_tested', []))})
- **Datasets Tested:** {len(summary.get('datasets_tested', []))} ({', '.join(summary.get('datasets_tested', []))})

---

## 🏆 Best Performing Models

| Rank | Model | Dataset | Accuracy | F1 Score | AUC | Training Time (s) |
|------|-------|---------|----------|----------|-----|-------------------|
"""
        
        # Add top 10 experiments
        top_experiments = df.head(10)
        for idx, row in top_experiments.iterrows():
            md_content += f"| {idx + 1} | **{row.get('model_name', 'N/A')}** | {row.get('dataset_type', 'N/A')} | {row.get('accuracy', 0):.4f} | {row.get('f1', 0):.4f} | {row.get('auc', 0):.4f} | {row.get('training_time', 0):.2f} |\n"
        
        md_content += "\n---\n\n## 📈 Performance by Model\n\n"
        md_content += "| Model | Avg Accuracy | Avg F1 | Best Dataset |\n"
        md_content += "|-------|--------------|--------|-------------|\n"
        
        # Group by model
        if 'model_name' in df.columns and 'accuracy' in df.columns:
            model_stats = df.groupby('model_name').agg({
                'accuracy': 'mean',
                'f1': 'mean'
            }).round(4)
            
            for model, stats in model_stats.iterrows():
                best_dataset = df[df['model_name'] == model].nlargest(1, 'accuracy')['dataset_type'].values[0] if len(df[df['model_name'] == model]) > 0 else 'N/A'
                md_content += f"| **{model}** | {stats['accuracy']:.4f} | {stats['f1']:.4f} | {best_dataset} |\n"
        
        md_content += "\n---\n\n*Report generated by Experiment Manager*  \n*Heart Disease Diagnosis Project - AIO2025*\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def export_to_csv(self, filename=None):
        """
        Export all experiments to CSV
        
        Args:
            filename (str): Output filename
            
        Returns:
            str: Path to CSV file
        """
        if not self.experiments:
            self.logger.warning("No experiments to export")
            return None
        
        if filename is None:
            filename = f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        output_file = self.results_dir / filename
        df = self.compare_experiments()
        df.to_csv(output_file, index=False)
        
        self.logger.info(f"Exported {len(df)} experiments to {output_file}")
        return str(output_file)


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = ExperimentManager()
    
    # Set global seed
    manager.set_global_seed(42)
    
    # Example: Log một experiment
    experiment_id = manager.log_experiment(
        model_name="xgboost",
        dataset_type="raw",
        params={
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1
        },
        metrics={
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.87,
            "f1": 0.85,
            "auc": 0.92
        },
        training_time=12.5
    )
    
    print(f"Logged experiment ID: {experiment_id}")
    
    # Compare experiments
    comparison = manager.compare_experiments(sort_by='accuracy')
    print("\nExperiment Comparison:")
    print(comparison.head())
    
    # Get best experiment
    best = manager.get_best_experiment(metric='accuracy')
    print(f"\nBest Experiment: {best}")
    
    # Generate report
    report_path = manager.generate_report(output_format='html')
    print(f"\nReport generated: {report_path}")
    
    # Export to CSV
    csv_path = manager.export_to_csv()
    print(f"Exported to CSV: {csv_path}")
