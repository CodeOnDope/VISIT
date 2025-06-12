# src/utils/alert_system.py
import logging
import time
import threading
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
import requests

from src.utils.analytics_db import AnalyticsDB
from datetime import datetime, timedelta
logger = logging.getLogger("AlertSystem")

class AlertRule:
    """Represents an alert rule for the system."""
    
    def __init__(self, rule_id, name, condition, threshold, message, enabled=True):
        self.rule_id = rule_id
        self.name = name
        self.condition = condition  # "gt" (greater than), "lt" (less than), etc.
        self.threshold = threshold
        self.message = message
        self.enabled = enabled
        self.last_triggered = None
        self.cooldown_minutes = 15  # Time before the same alert can trigger again

class AlertSystem:
    """System for generating alerts based on analytics data."""
    
    def __init__(self, config=None):
        """Initialize the alert system."""
        self.config = config or {}
        self.db = AnalyticsDB()
        self.rules = []
        self.load_rules()
        
        # Alert delivery methods
        self.notification_methods = []
        if self.config.get("email_enabled", False):
            self.notification_methods.append(self._send_email_alert)
        if self.config.get("sms_enabled", False):
            self.notification_methods.append(self._send_sms_alert)
        if self.config.get("screen_enabled", True):
            self.notification_methods.append(self._display_screen_alert)
        
        # Start monitoring thread
        self.running = False
        self.alert_thread = None
    
    def load_rules(self):
        """Load alert rules from configuration."""
        rules_config = self.config.get("rules", [])
        for rule_config in rules_config:
            self.rules.append(AlertRule(
                rule_id=rule_config.get("id", f"rule_{len(self.rules)}"),
                name=rule_config.get("name", "Unnamed Rule"),
                condition=rule_config.get("condition", "gt"),
                threshold=rule_config.get("threshold", 0),
                message=rule_config.get("message", "Alert triggered"),
                enabled=rule_config.get("enabled", True)
            ))
        
        # Add default rules if none are configured
        if not self.rules:
            self.rules.append(AlertRule(
                rule_id="congestion_alert",
                name="Congestion Alert",
                condition="gt",
                threshold=10,
                message="High visitor congestion detected in zone: {zone_id}",
                enabled=True
            ))
            
            self.rules.append(AlertRule(
                rule_id="low_engagement_alert",
                name="Low Engagement Alert",
                condition="lt",
                threshold=30,
                message="Low engagement detected at exhibit: {exhibit_id}",
                enabled=True
            ))
    
    def start_monitoring(self):
        """Start the alert monitoring system."""
        if self.running:
            return
            
        self.running = True
        self.alert_thread = threading.Thread(target=self._monitor_loop)
        self.alert_thread.daemon = True
        self.alert_thread.start()
        logger.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop the alert monitoring system."""
        self.running = False
        if self.alert_thread:
            self.alert_thread.join(timeout=1.0)
        logger.info("Alert monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop to check alert conditions."""
        while self.running:
            try:
                # Check each rule
                for rule in self.rules:
                    if not rule.enabled:
                        continue
                        
                    # Skip if in cooldown period
                    if rule.last_triggered and (datetime.now() - rule.last_triggered).total_seconds() < (rule.cooldown_minutes * 60):
                        continue
                    
                    # Check rule conditions
                    triggered = False
                    alert_data = {}
                    
                    if rule.rule_id == "congestion_alert":
                        # Check for congestion in zones
                        zones = self._get_zone_congestion()
                        for zone_id, count in zones.items():
                            if self._check_condition(count, rule.condition, rule.threshold):
                                triggered = True
                                alert_data = {"zone_id": zone_id, "visitor_count": count}
                                break
                    
                    elif rule.rule_id == "low_engagement_alert":
                        # Check for low engagement at exhibits
                        exhibits = self._get_exhibit_engagement()
                        for exhibit_id, dwell_time in exhibits.items():
                            if self._check_condition(dwell_time, rule.condition, rule.threshold):
                                triggered = True
                                alert_data = {"exhibit_id": exhibit_id, "dwell_time": dwell_time}
                                break
                    
                    # Trigger alert if condition met
                    if triggered:
                        self._trigger_alert(rule, alert_data)
                        rule.last_triggered = datetime.now()
            
            except Exception as e:
                logger.error(f"Error in alert monitoring: {e}")
            
            # Sleep before next check
            time.sleep(60)  # Check every minute
    
    def _get_zone_congestion(self):
        """Get current congestion levels for all zones."""
        try:
            conn = self.db.conn
            if conn:
                # Use a short time window (e.g., last 5 minutes)
                time_window = datetime.now() - timedelta(minutes=5)
                time_str = time_window.isoformat()
                
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT zone_id, AVG(visitor_count) FROM zone_activity WHERE timestamp >= ? GROUP BY zone_id",
                    (time_str,)
                )
                
                results = {}
                for zone_id, avg_count in cursor.fetchall():
                    results[zone_id] = avg_count
                
                return results
        except Exception as e:
            logger.error(f"Error getting zone congestion: {e}")
        
        return {}
    
    def _get_exhibit_engagement(self):
        """Get current engagement levels for all exhibits."""
        try:
            conn = self.db.conn
            if conn:
                # Use a short time window (e.g., last 5 minutes)
                time_window = datetime.now() - timedelta(minutes=5)
                time_str = time_window.isoformat()
                
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT exhibit_id, AVG(duration_ms) FROM exhibit_interactions WHERE timestamp >= ? GROUP BY exhibit_id",
                    (time_str,)
                )
                
                results = {}
                for exhibit_id, avg_duration in cursor.fetchall():
                    results[exhibit_id] = avg_duration / 1000  # Convert to seconds
                
                return results
        except Exception as e:
            logger.error(f"Error getting exhibit engagement: {e}")
        
        return {}
    
    def _check_condition(self, value, condition, threshold):
        """Check if a value meets a condition."""
        if condition == "gt":
            return value > threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "eq":
            return value == threshold
        elif condition == "neq":
            return value != threshold
        elif condition == "gte":
            return value >= threshold
        elif condition == "lte":
            return value <= threshold
        
        return False
    
    def _trigger_alert(self, rule, alert_data):
        """Trigger an alert based on the rule."""
        # Format alert message
        message = rule.message.format(**alert_data)
        alert_info = {
            "rule_id": rule.rule_id,
            "rule_name": rule.name,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "data": alert_data
        }
        
        logger.info(f"Alert triggered: {message}")
        
        # Send notifications through configured methods
        for method in self.notification_methods:
            try:
                method(alert_info)
            except Exception as e:
                logger.error(f"Error sending alert notification: {e}")
    
    def _send_email_alert(self, alert_info):
        """Send an alert via email."""
        try:
            # Email configuration
            email_config = self.config.get("email", {})
            smtp_server = email_config.get("smtp_server", "smtp.gmail.com")
            smtp_port = email_config.get("smtp_port", 587)
            sender_email = email_config.get("sender_email", "")
            sender_password = email_config.get("sender_password", "")
            recipient_emails = email_config.get("recipient_emails", [])
            
            if not sender_email or not sender_password or not recipient_emails:
                logger.warning("Email configuration incomplete, skipping email alert")
                return
            
            # Create message
            subject = f"VISIT Alert: {alert_info['rule_name']}"
            body = (f"Alert: {alert_info['message']}\n\n"
                   f"Triggered at: {alert_info['timestamp']}\n"
                   f"Rule: {alert_info['rule_name']} ({alert_info['rule_id']})")
            
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = sender_email
            msg['To'] = ", ".join(recipient_emails)
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent to {', '.join(recipient_emails)}")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
    
    def _send_sms_alert(self, alert_info):
        """Send an alert via SMS."""
        try:
            # SMS configuration
            sms_config = self.config.get("sms", {})
            api_key = sms_config.get("api_key", "")
            api_url = sms_config.get("api_url", "")
            phone_numbers = sms_config.get("phone_numbers", [])
            
            if not api_key or not api_url or not phone_numbers:
                logger.warning("SMS configuration incomplete, skipping SMS alert")
                return
            
            # Create message
            message = f"VISIT Alert: {alert_info['message']}"
            
            # Send SMS to each number
            for phone in phone_numbers:
                data = {
                    "api_key": api_key,
                    "to": phone,
                    "message": message
                }
                
                response = requests.post(api_url, json=data)
                if response.status_code == 200:
                    logger.info(f"SMS alert sent to {phone}")
                else:
                    logger.warning(f"Failed to send SMS alert to {phone}: {response.status_code}")
            
        except Exception as e:
            logger.error(f"Error sending SMS alert: {e}")
    
    def _display_screen_alert(self, alert_info):
        """Display an alert on the screen."""
        # Emit a signal or event that the UI can listen for
        # This would be implemented by connecting to your UI framework
        
        # For now, just log the alert
        logger.info(f"Screen alert: {alert_info['message']}")
        
        # In a real implementation, you might use QT signals:
        # if hasattr(self, 'alert_signal'):
        #     self.alert_signal.emit(alert_info)