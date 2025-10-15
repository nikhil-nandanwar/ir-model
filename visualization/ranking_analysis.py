"""
Visualization tools for analyzing and demonstrating the context-aware
information retrieval system's behavior.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from wordcloud import WordCloud
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Some visualizations will be limited.")

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class IRVisualizationTools:
    """Tools for visualizing IR system behavior and results."""
    
    def __init__(self, output_dir: str = "visualization/outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_ranking_comparison(self, query: str, baseline_results: List[Dict], 
                              personalized_results: List[Dict], 
                              save_path: str = None) -> str:
        """
        Compare baseline vs personalized ranking results.
        
        Args:
            query: Search query
            baseline_results: Results without personalization
            personalized_results: Results with personalization
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Prepare data
        baseline_titles = [r['title'][:30] + "..." if len(r['title']) > 30 else r['title'] 
                          for r in baseline_results[:10]]
        baseline_scores = [r['score'] for r in baseline_results[:10]]
        
        personalized_titles = [r['title'][:30] + "..." if len(r['title']) > 30 else r['title'] 
                             for r in personalized_results[:10]]
        personalized_scores = [r['score'] for r in personalized_results[:10]]
        
        # Baseline results
        y_pos1 = range(len(baseline_titles))
        bars1 = ax1.barh(y_pos1, baseline_scores, alpha=0.7, color='skyblue')
        ax1.set_yticks(y_pos1)
        ax1.set_yticklabels(baseline_titles)
        ax1.set_xlabel('Relevance Score')
        ax1.set_title(f'Baseline Results\nQuery: "{query}"')
        ax1.grid(True, alpha=0.3)
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars1, baseline_scores)):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', ha='left', va='center', fontsize=8)
        
        # Personalized results
        y_pos2 = range(len(personalized_titles))
        bars2 = ax2.barh(y_pos2, personalized_scores, alpha=0.7, color='lightcoral')
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(personalized_titles)
        ax2.set_xlabel('Relevance Score')
        ax2.set_title(f'Personalized Results\nQuery: "{query}"')
        ax2.grid(True, alpha=0.3)
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars2, personalized_scores)):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'ranking_comparison_{query.replace(" ", "_")}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_user_interest_evolution(self, user_profile_data: Dict, 
                                   save_path: str = None) -> str:
        """
        Plot how user interests evolve over time.
        
        Args:
            user_profile_data: User profile statistics over time
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data
        categories = user_profile_data.get('categories', {})
        keywords = user_profile_data.get('keywords', {})
        search_history = user_profile_data.get('search_history', [])
        
        # Plot 1: Category interests
        if categories:
            cat_names = list(categories.keys())[:8]  # Top 8 categories
            cat_scores = [categories[cat] for cat in cat_names]
            
            colors = plt.cm.Set3(range(len(cat_names)))
            wedges, texts, autotexts = ax1.pie(cat_scores, labels=cat_names, autopct='%1.1f%%',
                                              colors=colors, startangle=90)
            ax1.set_title('Interest Categories Distribution')
        
        # Plot 2: Top keywords
        if keywords:
            top_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:10]
            keyword_names = [k[0] for k in top_keywords]
            keyword_scores = [k[1] for k in top_keywords]
            
            bars = ax2.bar(range(len(keyword_names)), keyword_scores, alpha=0.7, color='lightgreen')
            ax2.set_xticks(range(len(keyword_names)))
            ax2.set_xticklabels(keyword_names, rotation=45, ha='right')
            ax2.set_ylabel('Interest Score')
            ax2.set_title('Top Keywords by Interest')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Search frequency over time
        if search_history:
            # Group searches by day
            daily_searches = {}
            for entry in search_history[-30:]:  # Last 30 searches
                if isinstance(entry, dict) and 'timestamp' in entry:
                    timestamp = datetime.fromisoformat(entry['timestamp'])
                    date_key = timestamp.date()
                    daily_searches[date_key] = daily_searches.get(date_key, 0) + 1
            
            if daily_searches:
                dates = sorted(daily_searches.keys())
                counts = [daily_searches[date] for date in dates]
                
                ax3.plot(dates, counts, marker='o', linewidth=2, markersize=6)
                ax3.set_xlabel('Date')
                ax3.set_ylabel('Number of Searches')
                ax3.set_title('Search Activity Over Time')
                ax3.grid(True, alpha=0.3)
                plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 4: Search query length distribution
        if search_history:
            query_lengths = []
            for entry in search_history:
                if isinstance(entry, dict) and 'query' in entry:
                    query_lengths.append(len(entry['query'].split()))
            
            if query_lengths:
                ax4.hist(query_lengths, bins=10, alpha=0.7, color='orange', edgecolor='black')
                ax4.set_xlabel('Query Length (words)')
                ax4.set_ylabel('Frequency')
                ax4.set_title('Query Length Distribution')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'user_interest_evolution.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_score_components(self, search_results: List[Dict], 
                            save_path: str = None) -> str:
        """
        Plot the contribution of different scoring components.
        
        Args:
            search_results: List of search results with score components
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        if not search_results:
            return None
        
        # Extract score components
        component_names = []
        if search_results[0].get('score_components'):
            component_names = list(search_results[0]['score_components'].keys())
        
        if not component_names:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prepare data for top 5 results
        top_results = search_results[:5]
        doc_titles = [r['title'][:20] + "..." if len(r['title']) > 20 else r['title'] 
                     for r in top_results]
        
        # Stacked bar chart showing component contributions
        component_data = {}
        for component in component_names:
            component_data[component] = [r['score_components'].get(component, 0) 
                                       for r in top_results]
        
        bottom = [0] * len(top_results)
        colors = plt.cm.Set2(range(len(component_names)))
        
        for i, (component, values) in enumerate(component_data.items()):
            ax1.bar(doc_titles, values, bottom=bottom, label=component, 
                   color=colors[i], alpha=0.8)
            bottom = [b + v for b, v in zip(bottom, values)]
        
        ax1.set_ylabel('Score Contribution')
        ax1.set_title('Score Components by Document')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Average component contribution across all results
        avg_components = {}
        for component in component_names:
            scores = [r['score_components'].get(component, 0) for r in search_results]
            avg_components[component] = sum(scores) / len(scores) if scores else 0
        
        components = list(avg_components.keys())
        values = list(avg_components.values())
        
        bars = ax2.bar(components, values, color=colors[:len(components)], alpha=0.8)
        ax2.set_ylabel('Average Score')
        ax2.set_title('Average Component Contributions')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'score_components.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_user_profile_wordcloud(self, user_profile_data: Dict, 
                                    save_path: str = None) -> str:
        """
        Create a word cloud from user's search interests.
        
        Args:
            user_profile_data: User profile data
            save_path: Path to save the word cloud
            
        Returns:
            Path to saved word cloud
        """
        try:
            from wordcloud import WordCloud
        except ImportError:
            print("Warning: WordCloud not available")
            return None
        
        # Combine keywords and categories with their weights
        text_freq = {}
        
        # Add keywords
        keywords = user_profile_data.get('keywords', {})
        for word, freq in keywords.items():
            text_freq[word] = freq
        
        # Add categories with higher weight
        categories = user_profile_data.get('categories', {})
        for category, freq in categories.items():
            text_freq[category] = freq * 2  # Give categories more weight
        
        if not text_freq:
            return None
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=50
        ).generate_from_frequencies(text_freq)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('User Interest Profile Word Cloud', fontsize=16, pad=20)
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'user_wordcloud.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_temporal_weighting_effect(self, search_history: List[Dict], 
                                     save_path: str = None) -> str:
        """
        Visualize how temporal weighting affects search history influence.
        
        Args:
            search_history: List of search history entries
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        if not search_history:
            return None
        
        from ..src.utils import TemporalWeighting
        
        temporal = TemporalWeighting()
        current_time = datetime.now()
        
        # Calculate weights for each search
        time_diffs = []
        weights_exp = []
        weights_linear = []
        queries = []
        
        for entry in search_history[-20:]:  # Last 20 searches
            if isinstance(entry, dict) and 'timestamp' in entry:
                timestamp = datetime.fromisoformat(entry['timestamp'])
                time_diff_hours = (current_time - timestamp).total_seconds() / 3600
                
                exp_weight = temporal.exponential_decay(time_diff_hours)
                linear_weight = temporal.linear_decay(time_diff_hours)
                
                time_diffs.append(time_diff_hours)
                weights_exp.append(exp_weight)
                weights_linear.append(linear_weight)
                queries.append(entry.get('query', ''))
        
        if not time_diffs:
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Weight decay over time
        time_range = range(0, int(max(time_diffs)) + 1, 6)  # Every 6 hours
        exp_decay_curve = [temporal.exponential_decay(t) for t in time_range]
        linear_decay_curve = [temporal.linear_decay(t) for t in time_range]
        
        ax1.plot(time_range, exp_decay_curve, 'b-', label='Exponential Decay', linewidth=2)
        ax1.plot(time_range, linear_decay_curve, 'r--', label='Linear Decay', linewidth=2)
        ax1.scatter(time_diffs, weights_exp, color='blue', alpha=0.6, s=50, label='Actual Searches (Exp)')
        ax1.scatter(time_diffs, weights_linear, color='red', alpha=0.6, s=50, label='Actual Searches (Linear)')
        
        ax1.set_xlabel('Hours Ago')
        ax1.set_ylabel('Weight')
        ax1.set_title('Temporal Weighting Functions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Individual search weights
        indices = range(len(queries))
        ax2.bar(indices, weights_exp, alpha=0.7, color='skyblue', label='Exponential Weight')
        ax2.set_xlabel('Search (chronological order)')
        ax2.set_ylabel('Weight')
        ax2.set_title('Individual Search Weights (Most Recent Searches)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add query labels (rotated)
        if len(queries) <= 10:  # Only show labels if not too many
            ax2.set_xticks(indices)
            ax2.set_xticklabels([q[:15] + "..." if len(q) > 15 else q for q in queries], 
                               rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'temporal_weighting.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_interactive_dashboard(self, search_results: List[Dict], 
                                   user_profile_data: Dict,
                                   save_path: str = None) -> Optional[str]:
        """
        Create an interactive Plotly dashboard (if available).
        
        Args:
            search_results: Search results to visualize
            user_profile_data: User profile data
            save_path: Path to save the HTML file
            
        Returns:
            Path to saved HTML file or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available for interactive dashboard")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Search Results Scores', 'User Categories', 
                          'Score Components', 'Search Activity'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Plot 1: Search results scores
        if search_results:
            top_results = search_results[:10]
            titles = [r['title'][:30] for r in top_results]
            scores = [r['score'] for r in top_results]
            
            fig.add_trace(
                go.Bar(x=titles, y=scores, name="Relevance Scores"),
                row=1, col=1
            )
        
        # Plot 2: User categories pie chart
        categories = user_profile_data.get('categories', {})
        if categories:
            fig.add_trace(
                go.Pie(labels=list(categories.keys())[:8], 
                      values=list(categories.values())[:8],
                      name="Categories"),
                row=1, col=2
            )
        
        # Plot 3: Score components for top result
        if search_results and search_results[0].get('score_components'):
            components = search_results[0]['score_components']
            fig.add_trace(
                go.Bar(x=list(components.keys()), y=list(components.values()),
                      name="Score Components"),
                row=2, col=1
            )
        
        # Plot 4: Search activity over time (mock data for demo)
        dates = [datetime.now() - timedelta(days=i) for i in range(7, 0, -1)]
        activity = [5, 8, 3, 12, 7, 9, 6]  # Mock activity data
        
        fig.add_trace(
            go.Scatter(x=dates, y=activity, mode='lines+markers',
                      name="Daily Searches"),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Information Retrieval System Dashboard",
            title_x=0.5
        )
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'interactive_dashboard.html')
        
        fig.write_html(save_path)
        
        return save_path
    
    def generate_comparison_report(self, baseline_results: List[Dict],
                                 personalized_results: List[Dict],
                                 query: str, user_id: str) -> str:
        """
        Generate a comprehensive visualization report.
        
        Args:
            baseline_results: Results without personalization
            personalized_results: Results with personalization
            query: Search query
            user_id: User identifier
            
        Returns:
            Path to report directory
        """
        report_dir = os.path.join(self.output_dir, f'report_{user_id}_{query.replace(" ", "_")}')
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate individual plots
        plots_generated = []
        
        # 1. Ranking comparison
        comparison_plot = self.plot_ranking_comparison(
            query, baseline_results, personalized_results,
            os.path.join(report_dir, 'ranking_comparison.png')
        )
        if comparison_plot:
            plots_generated.append(('Ranking Comparison', comparison_plot))
        
        # 2. Score components
        if personalized_results:
            components_plot = self.plot_score_components(
                personalized_results,
                os.path.join(report_dir, 'score_components.png')
            )
            if components_plot:
                plots_generated.append(('Score Components', components_plot))
        
        # 3. Interactive dashboard (if available)
        if PLOTLY_AVAILABLE and personalized_results:
            dashboard_path = self.create_interactive_dashboard(
                personalized_results, {},
                os.path.join(report_dir, 'dashboard.html')
            )
            if dashboard_path:
                plots_generated.append(('Interactive Dashboard', dashboard_path))
        
        # Generate summary HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>IR System Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #333; }}
                .section {{ margin: 30px 0; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                .stats {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Information Retrieval System Analysis Report</h1>
            <div class="stats">
                <p><strong>Query:</strong> "{query}"</p>
                <p><strong>User ID:</strong> {user_id}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Baseline Results:</strong> {len(baseline_results)}</p>
                <p><strong>Personalized Results:</strong> {len(personalized_results)}</p>
            </div>
        """
        
        # Add plots to HTML
        for plot_title, plot_path in plots_generated:
            if plot_path.endswith('.html'):
                html_content += f"""
                <div class="section">
                    <h2>{plot_title}</h2>
                    <p><a href="{os.path.basename(plot_path)}" target="_blank">Open Interactive Dashboard</a></p>
                </div>
                """
            else:
                html_content += f"""
                <div class="section">
                    <h2>{plot_title}</h2>
                    <div class="plot">
                        <img src="{os.path.basename(plot_path)}" alt="{plot_title}">
                    </div>
                </div>
                """
        
        html_content += """
        </body>
        </html>
        """
        
        # Save HTML report
        report_path = os.path.join(report_dir, 'report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Report generated: {report_dir}")
        return report_dir