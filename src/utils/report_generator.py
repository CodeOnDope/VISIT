"""
Report generator module for the VISIT Museum Tracker system.

Generates daily, weekly, and monthly analytics reports from database data,
creates charts, saves HTML reports, and optionally emails them.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import logging
from datetime import datetime, timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import schedule

from src.utils.analytics_db import AnalyticsDB

logger = logging.getLogger("ReportGenerator")

class ReportGenerator:
    """Generates analytics reports from the database."""

    def __init__(self, config=None):
        """Initialize the report generator."""
        self.config = config or {}
        self.db = AnalyticsDB()
        self.output_dir = self.config.get("output_dir", "reports")
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_daily_report(self, date=None):
        """Generate a daily report for the specified date."""
        if date is None:
            date = datetime.now().date() - timedelta(days=1)  # Yesterday

        logger.info(f"Generating daily report for {date}")

        date_str = date.strftime("%Y-%m-%d")
        filename = os.path.join(self.output_dir, f"daily_report_{date_str}.html")

        start_time = datetime.combine(date, datetime.min.time()).isoformat()
        end_time = datetime.combine(date, datetime.max.time()).isoformat()

        visitor_data = self._query_visitor_counts(start_time, end_time)
        zone_data = self._query_zone_activity(start_time, end_time)
        expression_data = self._query_expression_data(start_time, end_time)

        html = self._generate_html_report(date_str, visitor_data, zone_data, expression_data)

        with open(filename, "w") as f:
            f.write(html)

        logger.info(f"Daily report saved to {filename}")

        if self.config.get("email_enabled", False):
            self._send_email_report(filename, f"Daily Analytics Report - {date_str}")

        return filename

    def _query_visitor_counts(self, start_time, end_time):
        """Query visitor count data from the database."""
        try:
            conn = self.db.conn
            if conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT timestamp, count FROM visitor_count WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
                    (start_time, end_time)
                )
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error querying visitor counts: {e}")
        return []

    def _query_zone_activity(self, start_time, end_time):
        """Query zone activity data from the database."""
        try:
            conn = self.db.conn
            if conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT timestamp, zone_id, visitor_count, dwell_time_ms FROM zone_activity WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
                    (start_time, end_time)
                )
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error querying zone activity: {e}")
        return []

    def _query_expression_data(self, start_time, end_time):
        """Query expression data from the database."""
        try:
            conn = self.db.conn
            if conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT timestamp, expression, count FROM expression_data WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
                    (start_time, end_time)
                )
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error querying expression data: {e}")
        return []

    def _generate_html_report(self, date_str, visitor_data, zone_data, expression_data):
        """Generate an HTML report from the data."""
        visitor_fig_path = os.path.join(self.output_dir, f"visitor_chart_{date_str}.png")
        zone_fig_path = os.path.join(self.output_dir, f"zone_chart_{date_str}.png")
        expression_fig_path = os.path.join(self.output_dir, f"expression_chart_{date_str}.png")

        self._create_visitor_chart(visitor_data, visitor_fig_path)
        self._create_zone_chart(zone_data, zone_fig_path)
        self._create_expression_chart(expression_data, expression_fig_path)

        visitor_stats = self._generate_visitor_stats(visitor_data)
        zone_stats = self._generate_zone_stats(zone_data)
        expression_stats = self._generate_expression_stats(expression_data)

        html = f"""
        <html>
        <head>
            <title>Daily Analytics Report - {date_str}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .section {{ margin-bottom: 30px; }}
                .chart {{ max-width: 100%; height: auto; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>VISIT Museum Tracker Analytics Report</h1>
            <h2>Date: {date_str}</h2>

            <div class="section">
                <h3>Visitor Statistics</h3>
                <p>Total visitors: {visitor_stats['total']}</p>
                <p>Peak visitor count: {visitor_stats['peak']} at {visitor_stats['peak_time']}</p>
                <p>Average visitors: {visitor_stats['average']:.2f}</p>
                <img src="visitor_chart_{date_str}.png" class="chart">
            </div>

            <div class="section">
                <h3>Zone Activity</h3>
                <table>
                    <tr>
                        <th>Zone</th>
                        <th>Total Visitors</th>
                        <th>Average Dwell Time (sec)</th>
                    </tr>
        """

        for zone, stats in zone_stats.items():
            if zone == 'most_popular':
                continue
            html += f"""
                    <tr>
                        <td>{zone}</td>
                        <td>{stats['total_visitors']}</td>
                        <td>{stats['avg_dwell_time']:.2f}</td>
                    </tr>
            """

        html += f"""
                </table>
                <img src="zone_chart_{date_str}.png" class="chart">
            </div>

            <div class="section">
                <h3>Expression Analysis</h3>
                <table>
                    <tr>
                        <th>Expression</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
        """

        for expression, stats in expression_stats.items():
            if expression == 'most_common':
                continue
            html += f"""
                    <tr>
                        <td>{expression}</td>
                        <td>{stats['count']}</td>
                        <td>{stats['percentage']:.2f}%</td>
                    </tr>
            """

        html += f"""
                </table>
                <img src="expression_chart_{date_str}.png" class="chart">
            </div>

            <div class="section">
                <h3>Summary</h3>
                <p>
                    This report provides an overview of visitor activity on {date_str}.
                    The museum had a total of {visitor_stats['total']} visitors, with a peak of
                    {visitor_stats['peak']} visitors at {visitor_stats['peak_time']}.
                </p>
                <p>
                    The most popular zone was {zone_stats.get('most_popular', {}).get('zone', 'N/A')} with
                    {zone_stats.get('most_popular', {}).get('visitors', 'N/A')} visitors.
                </p>
                <p>
                    The most common expression observed was {expression_stats.get('most_common', {}).get('expression', 'N/A')}
                    ({expression_stats.get('most_common', {}).get('percentage', 0):.2f}% of all expressions).
                </p>
            </div>

            <footer>
                <p>Generated by VISIT Museum Tracker on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </footer>
        </body>
        </html>
        """
        return html

    def _create_visitor_chart(self, visitor_data, filename):
        if not visitor_data:
            return

        df = pd.DataFrame(visitor_data, columns=['timestamp', 'count'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        plt.figure(figsize=(10, 6))
        plt.plot(df['timestamp'], df['count'], 'b-')
        plt.title('Visitor Count Over Time')
        plt.xlabel('Time')
        plt.ylabel('Number of Visitors')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def _create_zone_chart(self, zone_data, filename):
        if not zone_data:
            return

        df = pd.DataFrame(zone_data, columns=['timestamp', 'zone_id', 'visitor_count', 'dwell_time_ms'])
        zone_counts = df.groupby('zone_id')['visitor_count'].sum()

        plt.figure(figsize=(10, 6))
        zone_counts.plot(kind='bar', color='skyblue')
        plt.title('Total Visitors by Zone')
        plt.xlabel('Zone')
        plt.ylabel('Number of Visitors')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def _create_expression_chart(self, expression_data, filename):
        if not expression_data:
            return

        df = pd.DataFrame(expression_data, columns=['timestamp', 'expression', 'count'])
        expression_counts = df.groupby('expression')['count'].sum()

        plt.figure(figsize=(8, 8))
        plt.pie(expression_counts, labels=expression_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Expression Distribution')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def _generate_visitor_stats(self, visitor_data):
        if not visitor_data:
            return {
                'total': 0,
                'peak': 0,
                'peak_time': 'N/A',
                'average': 0
            }

        df = pd.DataFrame(visitor_data, columns=['timestamp', 'count'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        total = df['count'].sum()
        peak = df['count'].max()
        peak_time = df.loc[df['count'].idxmax(), 'timestamp'].strftime('%H:%M:%S')
        average = df['count'].mean()

        return {
            'total': total,
            'peak': peak,
            'peak_time': peak_time,
            'average': average
        }

    def _generate_zone_stats(self, zone_data):
        if not zone_data:
            return {}

        df = pd.DataFrame(zone_data, columns=['timestamp', 'zone_id', 'visitor_count', 'dwell_time_ms'])

        zone_stats = {}
        for zone_id, group in df.groupby('zone_id'):
            total_visitors = group['visitor_count'].sum()
            avg_dwell_time = group['dwell_time_ms'].mean() / 1000 if len(group) > 0 else 0

            zone_stats[zone_id] = {
                'total_visitors': total_visitors,
                'avg_dwell_time': avg_dwell_time
            }

        if zone_stats:
            most_popular = max(zone_stats.items(), key=lambda x: x[1]['total_visitors'])
            zone_stats['most_popular'] = {
                'zone': most_popular[0],
                'visitors': most_popular[1]['total_visitors']
            }

        return zone_stats

    def _generate_expression_stats(self, expression_data):
        if not expression_data:
            return {}

        df = pd.DataFrame(expression_data, columns=['timestamp', 'expression', 'count'])

        expression_counts = df.groupby('expression')['count'].sum()
        total_count = expression_counts.sum()

        expression_stats = {}
        for expression, count in expression_counts.items():
            percentage = (count / total_count * 100) if total_count > 0 else 0

            expression_stats[expression] = {
                'count': count,
                'percentage': percentage
            }

        if expression_stats:
            most_common = max(expression_stats.items(), key=lambda x: x[1]['count'])
            expression_stats['most_common'] = {
                'expression': most_common[0],
                'count': most_common[1]['count'],
                'percentage': most_common[1]['percentage']
            }

        return expression_stats

    def _send_email_report(self, report_file, subject):
        try:
            email_config = self.config.get("email", {})
            smtp_server = email_config.get("smtp_server", "smtp.gmail.com")
            smtp_port = email_config.get("smtp_port", 587)
            sender_email = email_config.get("sender_email", "")
            sender_password = email_config.get("sender_password", "")
            recipient_emails = email_config.get("recipient_emails", [])

            if not sender_email or not sender_password or not recipient_emails:
                logger.warning("Email configuration incomplete, skipping email report")
                return

            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ", ".join(recipient_emails)
            msg['Subject'] = subject

            body = f"Please find attached the analytics report.\n\nGenerated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            msg.attach(MIMEText(body, 'plain'))

            with open(report_file, "rb") as f:
                report = MIMEApplication(f.read(), _subtype="html")
                report.add_header('Content-Disposition', 'attachment', filename=os.path.basename(report_file))
                msg.attach(report)

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)

            logger.info(f"Email report sent to {', '.join(recipient_emails)}")

        except Exception as e:
            logger.error(f"Error sending email report: {e}")

    def schedule_reports(self):
        """Schedule regular reports."""
        schedule.every().day.at("05:00").do(self.generate_daily_report)
        # Add weekly/monthly if implemented
        logger.info("Reports scheduled")

        while True:
            schedule.run_pending()
            time.sleep(60)
