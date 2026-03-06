"""
Utility Functions cho Heart Disease Diagnosis App
Các hàm hỗ trợ cho Streamlit app
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime
from pathlib import Path
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.platypus import Image as RLImage


class PatientHistoryManager:
    """Quản lý lịch sử dự đoán của bệnh nhân"""
    
    def __init__(self, history_file="data/patient_history.json"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history = self._load_history()
    
    def _load_history(self):
        """Load lịch sử từ file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_history(self):
        """Save lịch sử vào file"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def add_prediction(self, patient_id, patient_data, predictions, final_verdict):
        """
        Thêm một prediction vào history
        
        Args:
            patient_id (str): ID bệnh nhân
            patient_data (dict): Thông tin bệnh nhân
            predictions (list): Danh sách predictions từ các models
            final_verdict (str): Kết luận cuối cùng
        """
        record = {
            'patient_id': patient_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'patient_data': patient_data,
            'predictions': predictions,
            'final_verdict': final_verdict
        }
        
        self.history.append(record)
        self._save_history()
        return len(self.history) - 1
    
    def get_history(self, patient_id=None):
        """Lấy lịch sử (tất cả hoặc theo patient_id)"""
        if patient_id:
            return [h for h in self.history if h.get('patient_id') == patient_id]
        return self.history
    
    def get_statistics(self):
        """Tính toán thống kê từ history"""
        if not self.history:
            return {}
        
        df = pd.DataFrame(self.history)
        stats = {
            'total_predictions': len(df),
            'unique_patients': len(df['patient_id'].unique()) if 'patient_id' in df.columns else 0,
            'heart_disease_count': sum(1 for h in self.history if 'Heart Disease' in h.get('final_verdict', '')),
            'no_disease_count': sum(1 for h in self.history if 'No Heart Disease' in h.get('final_verdict', ''))
        }
        
        return stats


def create_patient_report_pdf(patient_data, predictions, final_verdict, contributions=None, output_file="patient_report.pdf"):
    """
    Tạo PDF report cho bệnh nhân
    
    Args:
        patient_data (dict): Thông tin bệnh nhân
        predictions (list): Danh sách predictions
        final_verdict (str): Kết luận cuối cùng
        output_file (str): Tên file output
    
    Returns:
        BytesIO: PDF file in memory
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1976d2'),
        spaceAfter=30,
        alignment=1  # Center
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#424242'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("🫀 Heart Disease Diagnosis Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"<b>Generated:</b> {timestamp}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Patient Information
    story.append(Paragraph("Patient Information", heading_style))
    patient_table_data = [['Feature', 'Value']]
    
    feature_names = {
        'age': 'Age (years)',
        'sex': 'Sex',
        'cp': 'Chest Pain Type',
        'trestbps': 'Resting Blood Pressure (mm Hg)',
        'chol': 'Cholesterol (mg/dl)',
        'fbs': 'Fasting Blood Sugar > 120 mg/dl',
        'restecg': 'Resting ECG',
        'thalach': 'Max Heart Rate',
        'exang': 'Exercise Induced Angina',
        'oldpeak': 'ST Depression',
        'slope': 'Slope of ST Segment',
        'ca': 'Number of Major Vessels',
        'thal': 'Thalassemia'
    }
    
    for key, value in patient_data.items():
        display_name = feature_names.get(key, key)
        patient_table_data.append([display_name, str(value)])
    
    patient_table = Table(patient_table_data, colWidths=[3*inch, 2*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(patient_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Final Verdict
    story.append(Paragraph("Diagnosis Result", heading_style))
    verdict_color = colors.HexColor('#d32f2f') if 'Heart Disease' in final_verdict else colors.HexColor('#388e3c')
    verdict_text = f'<font color="{verdict_color.hexval()}"><b>{final_verdict}</b></font>'
    story.append(Paragraph(verdict_text, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Model Predictions
    story.append(Paragraph("Individual Model Predictions", heading_style))
    pred_table_data = [['Model', 'Prediction', 'Confidence']]
    
    for pred in predictions:
        pred_table_data.append([
            pred['Model'],
            pred['Prediction'],
            f"{pred['Confidence']:.2%}"
        ])
    
    pred_table = Table(pred_table_data, colWidths=[2*inch, 2*inch, 1.5*inch])
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(pred_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Personalized Recommendations
    story.append(Paragraph("Personalized Recommendations", heading_style))
    
    if contributions:
        # Generate personalized recommendations based on contribution analysis
        feature_desc = get_feature_descriptions()
        recommendations_data = generate_personalized_recommendations(
            patient_data, contributions, feature_desc
        )
        
        recommendations_text = create_personalized_recommendations_text(
            recommendations_data, "Patient"
        )
        
        # Convert markdown-like text to paragraphs
        lines = recommendations_text.strip().split('\n')
        for line in lines:
            if line.strip():
                if line.startswith('🎯') or line.startswith('**') and line.endswith('**'):
                    # Header style for main sections
                    story.append(Paragraph(line.strip(), heading_style))
                elif line.startswith('**') and '.' in line:
                    # Bold for numbered recommendations
                    clean_line = line.replace('**', '<b>').replace('**', '</b>')
                    story.append(Paragraph(clean_line, styles['Normal']))
                    story.append(Spacer(1, 0.1*inch))
                elif line.startswith('   •') or line.startswith('•'):
                    # Bullet points
                    story.append(Paragraph(line.strip(), styles['Normal']))
                elif line.startswith('📋') or line.startswith('⚠️') or line.startswith('🏥'):
                    # Section headers with emojis
                    story.append(Spacer(1, 0.2*inch))
                    story.append(Paragraph(line.strip(), heading_style))
                else:
                    # Regular text
                    if line.strip():
                        story.append(Paragraph(line.strip(), styles['Normal']))
    else:
        # Fallback to general recommendations
        if 'Heart Disease' in final_verdict:
            recommendations = """
            • Consult with a cardiologist immediately
            • Monitor blood pressure and cholesterol levels regularly
            • Adopt a heart-healthy diet (low in saturated fats and sodium)
            • Engage in regular physical activity as recommended by your doctor
            • Avoid smoking and limit alcohol consumption
            • Manage stress through relaxation techniques
            """
        else:
            recommendations = """
            • Continue maintaining a healthy lifestyle
            • Regular health check-ups (annual cardiac screening)
            • Maintain a balanced diet and regular exercise
            • Monitor blood pressure and cholesterol levels
            • Avoid smoking and excessive alcohol consumption
            """
        
        story.append(Paragraph(recommendations, styles['Normal']))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Footer
    footer_text = """
    <i>Note: This report is generated by an AI-based system and should not replace professional medical advice. 
    Please consult with a qualified healthcare provider for proper diagnosis and treatment.</i>
    <br/><br/>
    <b>Team:</b> Dũng, Anh, Vinh, Hằng, Huy | AIO2025 VietAI
    """
    story.append(Paragraph(footer_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


def create_feature_importance_plot(model, feature_names, model_name):
    """
    Tạo biểu đồ feature importance
    
    Args:
        model: Trained model
        feature_names (list): Tên các features
        model_name (str): Tên model
    
    Returns:
        plotly.graph_objects.Figure
    """
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return None
    
    # Create dataframe
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    # Create plot
    fig = go.Figure(go.Bar(
        x=df['importance'],
        y=df['feature'],
        orientation='h',
        marker=dict(
            color=df['importance'],
            colorscale='Viridis',
            showscale=True
        )
    ))
    
    fig.update_layout(
        title=f'Feature Importance - {model_name}',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def calculate_input_contribution(model, input_data, feature_names, model_name, background_data=None):
    """
    Tính contribution của từng input field vào prediction hiện tại
    Sử dụng SHAP hoặc alternative methods cho different model types
    
    Args:
        model: Trained model
        input_data (pd.DataFrame): Input data (single row)
        feature_names (list): Tên các features
        model_name (str): Tên model
        background_data (np.ndarray): Background data for SHAP
    
    Returns:
        dict: {'feature': contribution_value, 'direction': 'positive/negative'}
    """
    try:
        # Import SHAP nếu có
        try:
            import shap
            has_shap = True
        except ImportError:
            has_shap = False
        
        # Convert input to numpy
        if isinstance(input_data, pd.DataFrame):
            X = input_data.values
        else:
            X = input_data
        
        contributions = {}
        
        if has_shap and hasattr(model, 'predict_proba'):
            try:
                # Sử dụng SHAP Explainer
                if background_data is not None:
                    explainer = shap.Explainer(model, background_data)
                else:
                    # Use a small sample as background nếu không có
                    explainer = shap.Explainer(model, X[:1])  # Use input itself as background
                
                # Calculate SHAP values
                shap_values = explainer(X)
                
                # Get SHAP values for prediction class
                if len(shap_values.shape) == 3:  # Multi-class
                    # Get values for positive class (class 1)
                    values = shap_values.values[0, :, 1]
                else:  # Binary classification
                    values = shap_values.values[0]
                
                # Create contributions dict
                for i, feature in enumerate(feature_names[:len(values)]):
                    contribution = float(values[i])
                    contributions[feature] = {
                        'value': abs(contribution),
                        'direction': 'positive' if contribution > 0 else 'negative',
                        'raw_contribution': contribution
                    }
                
                return contributions
                
            except Exception as e:
                print(f"SHAP failed for {model_name}: {str(e)}")
                # Fall back to alternative method
        
        # Alternative method: Use feature importance weighted by input values
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            input_values = X[0]
            
            # Calculate weighted contributions
            total_importance = np.sum(importance)
            for i, feature in enumerate(feature_names[:len(importance)]):
                # Normalize input value (simple min-max to 0-1)
                # This is a simplified approach
                normalized_value = min(max(input_values[i] / 100.0, 0), 1)  # Assuming max value ~100
                weighted_contribution = importance[i] * normalized_value
                
                contributions[feature] = {
                    'value': float(weighted_contribution),
                    'direction': 'positive',  # Simplified
                    'raw_contribution': float(weighted_contribution)
                }
        
        elif hasattr(model, 'coef_'):
            # For linear models, use coefficients * input values
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            input_values = X[0]
            
            for i, feature in enumerate(feature_names[:len(coef)]):
                contribution = float(coef[i] * input_values[i])
                contributions[feature] = {
                    'value': abs(contribution),
                    'direction': 'positive' if contribution > 0 else 'negative',
                    'raw_contribution': contribution
                }
        
        else:
            # For models without interpretable weights, return uniform contribution
            uniform_contribution = 1.0 / len(feature_names)
            for feature in feature_names:
                contributions[feature] = {
                    'value': uniform_contribution,
                    'direction': 'positive',
                    'raw_contribution': uniform_contribution
                }
        
        return contributions
        
    except Exception as e:
        print(f"Error calculating contributions for {model_name}: {str(e)}")
        return {}


def create_input_contribution_plot(contributions, input_data, model_name, prediction_result):
    """
    Tạo visualization cho input contribution dạng pie chart
    
    Args:
        contributions (dict): Contribution values từ calculate_input_contribution
        input_data (dict): Raw input data với field names và values
        model_name (str): Tên model
        prediction_result (str): "Heart Disease" or "No Heart Disease"
    
    Returns:
        plotly.graph_objects.Figure
    """
    if not contributions:
        return None
    
    # Prepare data cho visualization
    features = []
    contrib_values = []
    directions = []
    input_values = []
    raw_contribs = []
    
    for feature, contrib_info in contributions.items():
        features.append(feature)
        contrib_values.append(contrib_info['value'])
        directions.append(contrib_info['direction'])
        input_values.append(input_data.get(feature, 'N/A'))
        raw_contribs.append(contrib_info['raw_contribution'])
    
    # Convert to DataFrame và sort by contribution (descending)
    df = pd.DataFrame({
        'feature': features,
        'contribution': contrib_values,
        'direction': directions,
        'input_value': input_values,
        'raw_contribution': raw_contribs
    }).sort_values('contribution', ascending=False)
    
    # Calculate percentages (normalize to 100%)
    total_contribution = df['contribution'].sum()
    if total_contribution > 0:
        df['percentage'] = (df['contribution'] / total_contribution * 100).round(1)
    else:
        df['percentage'] = 100.0 / len(df)  # Equal distribution if all zeros
    
    # Create colors: gradient from red (high risk) to green (low risk)
    colors = []
    risk_colors = ['#8B0000', '#DC143C', '#FF6B6B', '#FFB6C1']  # Dark red to light red
    protective_colors = ['#90EE90', '#32CD32', '#228B22', '#006400']  # Light green to dark green
    neutral_colors = ['#FFD700', '#FFA500', '#FF8C00', '#FF7F50']  # Gold to orange
    
    for i, direction in enumerate(df['direction']):
        if direction == 'positive':  # Risk factors
            color_idx = min(i, len(risk_colors) - 1)
            colors.append(risk_colors[color_idx])
        elif direction == 'negative':  # Protective factors
            color_idx = min(i, len(protective_colors) - 1)
            colors.append(protective_colors[color_idx])
        else:  # Neutral
            color_idx = min(i, len(neutral_colors) - 1)
            colors.append(neutral_colors[color_idx])
    
    # Create labels with feature name, input value, and percentage
    labels = []
    for _, row in df.iterrows():
        feature_name = row['feature'].replace('_', ' ').title()
        labels.append(f"{feature_name}<br>Value: {row['input_value']}")
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=df['percentage'],
        hole=0.4,  # Donut chart
        marker=dict(
            colors=colors,
            line=dict(color='white', width=2)
        ),
        textinfo='label+percent',
        textposition='outside',
        hovertemplate=(
            "<b>%{label}</b><br>" +
            "Contribution: %{value:.1f}%<br>" +
            "Raw Score: %{customdata:.4f}<br>" +
            "<extra></extra>"
        ),
        customdata=df['raw_contribution'],
        textfont=dict(size=12)
    )])
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Input Contribution Analysis - {model_name}<br><sub>Prediction: {prediction_result}</sub>',
            x=0.5,
            font=dict(size=16, color='#333')
        ),
        height=700,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        ),
        margin=dict(l=20, r=150, t=80, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        annotations=[
            dict(
                text=f'<b>Total Features:<br>{len(df)}</b>',
                x=0.5, y=0.5,
                font=dict(size=14, color='#666'),
                showarrow=False
            )
        ]
    )
    
    return fig


def create_contribution_summary_table(contributions, input_data, feature_descriptions=None):
    """
    Tạo bảng summary cho input contributions
    
    Args:
        contributions (dict): Contribution values
        input_data (dict): Raw input data
        feature_descriptions (dict): Descriptions for features
    
    Returns:
        pd.DataFrame: Summary table
    """
    summary_data = []
    
    # Sort contributions by absolute value (descending)
    sorted_contrib = sorted(
        contributions.items(), 
        key=lambda x: abs(x[1]['raw_contribution']), 
        reverse=True
    )
    
    for feature, contrib_info in sorted_contrib:
        # Get feature description
        if feature_descriptions and feature in feature_descriptions:
            desc = feature_descriptions[feature]
            feature_name = desc['name']
            normal_range = desc.get('normal_range', '')
        else:
            feature_name = feature.replace('_', ' ').title()
            normal_range = ''
        
        summary_data.append({
            'Feature': feature_name,
            'Your Value': input_data.get(feature, 'N/A'),
            'Normal Range': normal_range,
            'Contribution': f"{contrib_info['raw_contribution']:+.4f}",
            'Impact': '🔴 Risk Factor' if contrib_info['direction'] == 'positive' else '🟢 Protective Factor',
            'Strength': f"{contrib_info['value']:.4f}"
        })
    
    return pd.DataFrame(summary_data)


def generate_personalized_recommendations(patient_data, contributions, feature_descriptions=None):
    """
    Tạo recommendations cá nhân hóa dựa trên feature importance và input của bệnh nhân
    
    Args:
        patient_data (dict): Input data của bệnh nhân
        contributions (dict): Contribution analysis từ model
        feature_descriptions (dict): Descriptions của features
    
    Returns:
        dict: {
            'top_risk_factors': list,
            'recommendations': list,
            'improvement_potential': dict
        }
    """
    if not contributions:
        return {
            'top_risk_factors': [],
            'recommendations': [],
            'improvement_potential': {}
        }
    
    # Sắp xếp contributions theo mức độ risk (positive contributions)
    risk_factors = []
    for feature, contrib_info in contributions.items():
        if contrib_info['raw_contribution'] > 0:  # Positive = risk factor
            risk_factors.append({
                'feature': feature,
                'contribution': contrib_info['raw_contribution'],
                'current_value': patient_data.get(feature, 'N/A'),
                'direction': contrib_info['direction']
            })
    
    # Sort theo contribution descending
    risk_factors.sort(key=lambda x: x['contribution'], reverse=True)
    
    # Lấy top 3 risk factors có thể cải thiện
    top_risk_factors = risk_factors[:3]
    
    # Feature ranges và recommendations
    feature_targets = {
        'age': {
            'optimal_range': 'N/A (unchangeable)',
            'recommendation': 'Age cannot be changed, but focus on other modifiable risk factors',
            'modifiable': False
        },
        'sex': {
            'optimal_range': 'N/A (unchangeable)', 
            'recommendation': 'Sex cannot be changed, but focus on other modifiable risk factors',
            'modifiable': False
        },
        'cp': {
            'optimal_range': '4 (Asymptomatic)',
            'recommendation': 'Work with your doctor to manage chest pain symptoms through medication and lifestyle changes',
            'modifiable': True
        },
        'trestbps': {
            'optimal_range': '<120 mm Hg',
            'recommendation': 'Reduce blood pressure through: low-sodium diet, regular exercise, stress management, and medication if needed',
            'modifiable': True
        },
        'chol': {
            'optimal_range': '<200 mg/dl',
            'recommendation': 'Lower cholesterol with: Mediterranean diet, reduce saturated fats, increase fiber, regular exercise, and statins if prescribed',
            'modifiable': True
        },
        'fbs': {
            'optimal_range': '0 (False - <120 mg/dl)',
            'recommendation': 'Control blood sugar through: balanced diet, regular meals, exercise, weight management, and diabetes medication if needed',
            'modifiable': True
        },
        'restecg': {
            'optimal_range': '0 (Normal)',
            'recommendation': 'Improve heart rhythm through: stress reduction, adequate sleep, limiting caffeine/alcohol, and following prescribed treatments',
            'modifiable': True
        },
        'thalach': {
            'optimal_range': 'Age-appropriate (220 - age)',
            'recommendation': 'Improve cardiovascular fitness through: gradual aerobic exercise, strength training, and cardiac rehabilitation if recommended',
            'modifiable': True
        },
        'exang': {
            'optimal_range': '0 (No)',
            'recommendation': 'Reduce exercise-induced angina through: gradual fitness improvement, proper warm-up, medication, and stress management',
            'modifiable': True
        },
        'oldpeak': {
            'optimal_range': '<1.0',
            'recommendation': 'Improve ST depression through: cardiac rehabilitation, regular exercise, medication compliance, and stress management',
            'modifiable': True
        },
        'slope': {
            'optimal_range': '1 (Upsloping)',
            'recommendation': 'Improve exercise response through: cardiac rehabilitation, gradual fitness improvement, and medical management',
            'modifiable': True
        },
        'ca': {
            'optimal_range': '0 (No blockages)',
            'recommendation': 'Prevent further vessel blockage through: heart-healthy diet, exercise, medication, and possible cardiac procedures',
            'modifiable': True
        },
        'thal': {
            'optimal_range': '3 (Normal)',
            'recommendation': 'Manage thalassemia effects through: regular monitoring, appropriate treatments, and cardiovascular protection',
            'modifiable': True
        }
    }
    
    # Generate specific recommendations
    recommendations = []
    improvement_potential = {}
    
    for i, risk_factor in enumerate(top_risk_factors, 1):
        feature = risk_factor['feature']
        current_value = risk_factor['current_value']
        contribution = risk_factor['contribution']
        
        if feature in feature_targets:
            target_info = feature_targets[feature]
            
            if target_info['modifiable']:
                # Get feature display name
                if feature_descriptions and feature in feature_descriptions:
                    display_name = feature_descriptions[feature]['name']
                else:
                    display_name = feature.replace('_', ' ').title()
                
                recommendation = {
                    'rank': i,
                    'feature': display_name,
                    'current_value': current_value,
                    'target_range': target_info['optimal_range'],
                    'recommendation': target_info['recommendation'],
                    'risk_contribution': f"{contribution:.4f}",
                    'priority': 'High' if i == 1 else 'Medium' if i == 2 else 'Low'
                }
                
                recommendations.append(recommendation)
                improvement_potential[feature] = {
                    'current': current_value,
                    'target': target_info['optimal_range'],
                    'potential_reduction': contribution * 0.7  # Estimate 70% improvement possible
                }
    
    return {
        'top_risk_factors': top_risk_factors,
        'recommendations': recommendations,
        'improvement_potential': improvement_potential
    }


def create_personalized_recommendations_text(recommendations_data, patient_name="Patient"):
    """
    Tạo text recommendations cho PDF và display
    
    Args:
        recommendations_data (dict): Data từ generate_personalized_recommendations
        patient_name (str): Tên bệnh nhân
    
    Returns:
        str: Formatted recommendations text
    """
    if not recommendations_data['recommendations']:
        return """
        Based on your current health profile, continue maintaining your healthy lifestyle:
        • Regular health check-ups (annual cardiac screening)
        • Maintain a balanced diet and regular exercise
        • Monitor blood pressure and cholesterol levels
        • Avoid smoking and excessive alcohol consumption
        """
    
    recommendations = recommendations_data['recommendations']
    
    text = f"""
🎯 **Personalized Action Plan for {patient_name}**

Based on your input data and AI analysis, here are the TOP 3 modifiable risk factors that would have the biggest impact on reducing your heart disease risk:

"""
    
    for rec in recommendations:
        text += f"""
**{rec['rank']}. {rec['feature']} - {rec['priority']} Priority**
   • Current Value: {rec['current_value']}
   • Target Range: {rec['target_range']}
   • Action Plan: {rec['recommendation']}
   • Risk Contribution: {float(rec['risk_contribution']):.3f}

"""
    
    text += """
📋 **Implementation Timeline:**
• Week 1-2: Consult with your healthcare provider about these specific areas
• Month 1: Begin implementing lifestyle changes for your top priority factor
• Month 2-3: Add interventions for your second and third priority factors
• Month 6: Follow-up assessment to measure improvements

⚠️ **Important Notes:**
• These recommendations are based on AI analysis and should complement, not replace, professional medical advice
• Work with your healthcare team to create a safe and effective improvement plan
• Regular monitoring is essential to track progress and adjust strategies

🏥 **Next Steps:**
1. Schedule appointment with your primary care physician
2. Share this analysis with your healthcare team
3. Consider referral to specialists (cardiologist, nutritionist) as needed
4. Begin with the highest priority recommendation first
"""
    
    return text


def calculate_model_agreement(predictions):
    """
    Tính % models đồng ý với nhau
    
    Args:
        predictions (list): Danh sách binary predictions (0 hoặc 1)
    
    Returns:
        dict: Agreement statistics
    """
    if not predictions:
        return {}
    
    total_models = len(predictions)
    disease_count = sum(predictions)
    no_disease_count = total_models - disease_count
    
    agreement_pct = max(disease_count, no_disease_count) / total_models * 100
    
    return {
        'total_models': total_models,
        'disease_votes': disease_count,
        'no_disease_votes': no_disease_count,
        'agreement_percentage': agreement_pct,
        'consensus': 'Strong' if agreement_pct >= 80 else 'Moderate' if agreement_pct >= 60 else 'Weak'
    }


def export_experiments_to_excel(experiments_df, output_file="experiments_summary.xlsx"):
    """
    Export experiments to Excel với multiple sheets
    
    Args:
        experiments_df (pd.DataFrame): DataFrame chứa experiments
        output_file (str): Tên file output
    
    Returns:
        BytesIO: Excel file in memory
    """
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Sheet 1: All experiments
        experiments_df.to_excel(writer, sheet_name='All Experiments', index=False)
        
        # Sheet 2: Summary by model
        if 'model_name' in experiments_df.columns and 'accuracy' in experiments_df.columns:
            model_summary = experiments_df.groupby('model_name').agg({
                'accuracy': ['mean', 'std', 'max', 'min'],
                'f1': ['mean', 'std', 'max', 'min']
            }).round(4)
            model_summary.to_excel(writer, sheet_name='Summary by Model')
        
        # Sheet 3: Summary by dataset
        if 'dataset_type' in experiments_df.columns:
            dataset_summary = experiments_df.groupby('dataset_type').agg({
                'accuracy': ['mean', 'std', 'max', 'min'],
                'f1': ['mean', 'std', 'max', 'min']
            }).round(4)
            dataset_summary.to_excel(writer, sheet_name='Summary by Dataset')
        
        # Sheet 4: Best configurations
        best_configs = experiments_df.nlargest(10, 'accuracy')[
            ['model_name', 'dataset_type', 'accuracy', 'f1', 'auc']
        ]
        best_configs.to_excel(writer, sheet_name='Top 10 Configurations', index=False)
    
    buffer.seek(0)
    return buffer


def get_feature_descriptions():
    """
    Trả về mô tả chi tiết cho các features
    
    Returns:
        dict: Feature descriptions
    """
    return {
        'age': {
            'name': 'Age',
            'description': 'Age of the patient in years',
            'normal_range': '29-77',
            'unit': 'years'
        },
        'sex': {
            'name': 'Sex',
            'description': 'Gender of the patient',
            'values': {'0': 'Female', '1': 'Male'}
        },
        'cp': {
            'name': 'Chest Pain Type',
            'description': 'Type of chest pain experienced',
            'values': {
                '1': 'Typical Angina',
                '2': 'Atypical Angina',
                '3': 'Non-anginal Pain',
                '4': 'Asymptomatic'
            }
        },
        'trestbps': {
            'name': 'Resting Blood Pressure',
            'description': 'Resting blood pressure (on admission to hospital)',
            'normal_range': '94-200',
            'unit': 'mm Hg'
        },
        'chol': {
            'name': 'Serum Cholesterol',
            'description': 'Serum cholesterol level',
            'normal_range': '126-564',
            'unit': 'mg/dl'
        },
        'fbs': {
            'name': 'Fasting Blood Sugar',
            'description': 'Fasting blood sugar > 120 mg/dl',
            'values': {'0': 'False', '1': 'True'}
        },
        'restecg': {
            'name': 'Resting ECG',
            'description': 'Resting electrocardiographic results',
            'values': {
                '0': 'Normal',
                '1': 'ST-T wave abnormality',
                '2': 'Left ventricular hypertrophy'
            }
        },
        'thalach': {
            'name': 'Maximum Heart Rate',
            'description': 'Maximum heart rate achieved',
            'normal_range': '71-202',
            'unit': 'bpm'
        },
        'exang': {
            'name': 'Exercise Induced Angina',
            'description': 'Exercise induced angina',
            'values': {'0': 'No', '1': 'Yes'}
        },
        'oldpeak': {
            'name': 'ST Depression',
            'description': 'ST depression induced by exercise relative to rest',
            'normal_range': '0-6.2',
            'unit': 'mm'
        },
        'slope': {
            'name': 'Slope of ST Segment',
            'description': 'The slope of the peak exercise ST segment',
            'values': {
                '1': 'Upsloping',
                '2': 'Flat',
                '3': 'Downsloping'
            }
        },
        'ca': {
            'name': 'Number of Major Vessels',
            'description': 'Number of major vessels colored by fluoroscopy',
            'values': {'0': '0', '1': '1', '2': '2', '3': '3'}
        },
        'thal': {
            'name': 'Thalassemia',
            'description': 'Thalassemia blood disorder',
            'values': {
                '3': 'Normal',
                '6': 'Fixed Defect',
                '7': 'Reversible Defect'
            }
        }
    }


def get_preset_examples():
    """
    Trả về các preset examples để test
    
    Returns:
        dict: Preset patient examples
    """
    return {
        'Normal Patient': {
            'age': 35,
            'sex': 0,
            'cp': 4,
            'trestbps': 115,
            'chol': 180,
            'fbs': 0,
            'restecg': 0,
            'thalach': 185,
            'exang': 0,
            'oldpeak': 0.0,
            'slope': 1,
            'ca': 0,
            'thal': 3
        },
        'Low Risk': {
            'age': 45,
            'sex': 0,
            'cp': 3,
            'trestbps': 125,
            'chol': 210,
            'fbs': 0,
            'restecg': 0,
            'thalach': 165,
            'exang': 0,
            'oldpeak': 0.3,
            'slope': 1,
            'ca': 0,
            'thal': 3
        },
        'Medium Risk': {
            'age': 55,
            'sex': 1,
            'cp': 2,
            'trestbps': 135,
            'chol': 245,
            'fbs': 1,
            'restecg': 1,
            'thalach': 145,
            'exang': 0,
            'oldpeak': 1.2,
            'slope': 2,
            'ca': 1,
            'thal': 6
        },
        'High Risk': {
            'age': 65,
            'sex': 1,
            'cp': 1,
            'trestbps': 155,
            'chol': 285,
            'fbs': 1,
            'restecg': 2,
            'thalach': 125,
            'exang': 1,
            'oldpeak': 2.5,
            'slope': 3,
            'ca': 2,
            'thal': 7
        }
    }


# Example usage
if __name__ == "__main__":
    # Test PatientHistoryManager
    history_manager = PatientHistoryManager()
    
    # Add a sample prediction
    sample_data = {'age': 50, 'sex': 1, 'cp': 2}
    sample_predictions = [
        {'Model': 'XGBoost', 'Prediction': 'Heart Disease', 'Confidence': 0.85},
        {'Model': 'Random Forest', 'Prediction': 'Heart Disease', 'Confidence': 0.78}
    ]
    
    history_manager.add_prediction('P001', sample_data, sample_predictions, 'Heart Disease')
    
    # Get statistics
    stats = history_manager.get_statistics()
    print("Statistics:", stats)
    
    print("\n✅ app_utils.py tested successfully!")
