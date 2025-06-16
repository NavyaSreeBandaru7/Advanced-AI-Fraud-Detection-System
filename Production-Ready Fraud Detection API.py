from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import time
import os
import sys
from typing import Dict, List, Any
import traceback
from functools import wraps
from datetime import datetime
import json

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fraud_detector import AdvancedFraudDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global fraud detector instance
fraud_detector = None
api_stats = {
    'requests_total': 0,
    'requests_successful': 0,
    'requests_failed': 0,
    'avg_response_time': 0,
    'start_time': datetime.now().isoformat()
}


def timing_decorator(f):
    """Decorator to measure API response times"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = f(*args, **kwargs)
            api_stats['requests_successful'] += 1
            return result
        except Exception as e:
            api_stats['requests_failed'] += 1
            raise
        finally:
            end_time = time.time()
            response_time = end_time - start_time
            api_stats['requests_total'] += 1
            
            # Update running average
            total_requests = api_stats['requests_total']
            current_avg = api_stats['avg_response_time']
            api_stats['avg_response_time'] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
    return wrapper


def validate_transaction_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate incoming transaction data
    
    Args:
        data: Transaction data dictionary
        
    Returns:
        Validated transaction data
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(data, dict):
        raise ValueError("Transaction data must be a dictionary")
    
    # Check for required fields (can be customized)
    required_fields = []  # Add required fields as needed
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate data types for numeric fields
    for key, value in data.items():
        if isinstance(value, str):
            try:
                # Try to convert string numbers to float
                data[key] = float(value)
            except ValueError:
                # Keep as string if conversion fails
                pass
    
    return data


@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(e)}\n{traceback.format_exc()}")
    
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred',
        'timestamp': datetime.now().isoformat(),
        'status': 'error'
    }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global fraud_detector
    
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'uptime_seconds': (datetime.now() - datetime.fromisoformat(api_stats['start_time'])).total_seconds(),
        'model_loaded': fraud_detector is not None and fraud_detector.is_trained,
        'api_stats': api_stats
    }
    
    if fraud_detector is None:
        health_status['status'] = 'degraded'
        health_status['message'] = 'Fraud detector not initialized'
        return jsonify(health_status), 503
    
    if not fraud_detector.is_trained:
        health_status['status'] = 'degraded'
        health_status['message'] = 'Fraud detector not trained'
        return jsonify(health_status), 503
    
    return jsonify(health_status), 200


@app.route('/api/v1/score', methods=['POST'])
@timing_decorator
def score_transaction():
    """
    Score a single transaction for fraud probability
    
    Request body:
    {
        "Amount": 150.00,
        "V1": -1.359807,
        "V2": -0.072781,
        ... (other features)
    }
    
    Response:
    {
        "fraud_probability": 0.234,
        "risk_level": "LOW",
        "recommendation": "APPROVE",
        "confidence": 0.876,
        "risk_factors": [],
        "model_scores": {...},
        "timestamp": "2024-01-01T12:00:00",
        "status": "success"
    }
    """
    global fraud_detector
    
    try:
        # Check if model is available
        if fraud_detector is None or not fraud_detector.is_trained:
            return jsonify({
                'error': 'Fraud detection model not available',
                'status': 'service_unavailable',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        # Get and validate request data
        if not request.is_json:
            return jsonify({
                'error': 'Request must be JSON',
                'status': 'bad_request',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        request_data = request.get_json()
        
        if 'transactions' not in request_data:
            return jsonify({
                'error': 'Missing transactions array',
                'status': 'bad_request',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        transactions = request_data['transactions']
        
        if not isinstance(transactions, list):
            return jsonify({
                'error': 'Transactions must be an array',
                'status': 'bad_request',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        if len(transactions) > 100:  # Limit batch size
            return jsonify({
                'error': 'Batch size too large (max 100 transactions)',
                'status': 'bad_request',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Process batch
        results = []
        for i, transaction in enumerate(transactions):
            try:
                validated_data = validate_transaction_data(transaction)
                result = fraud_detector.score_transaction(validated_data)
                result['transaction_index'] = i
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process transaction {i}: {str(e)}")
                results.append({
                    'transaction_index': i,
                    'error': str(e),
                    'status': 'failed',
                    'timestamp': datetime.now().isoformat()
                })
        
        # Calculate summary statistics
        successful_results = [r for r in results if 'fraud_probability' in r]
        summary = {
            'total_transactions': len(transactions),
            'successful_transactions': len(successful_results),
            'failed_transactions': len(transactions) - len(successful_results),
            'avg_fraud_probability': sum(r['fraud_probability'] for r in successful_results) / len(successful_results) if successful_results else 0,
            'high_risk_count': sum(1 for r in successful_results if r.get('risk_level') == 'HIGH'),
            'medium_risk_count': sum(1 for r in successful_results if r.get('risk_level') == 'MEDIUM'),
            'low_risk_count': sum(1 for r in successful_results if r.get('risk_level') in ['LOW', 'VERY_LOW'])
        }
        
        return jsonify({
            'batch_results': results,
            'summary': summary,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Batch scoring error: {str(e)}")
        return jsonify({
            'error': 'Batch scoring failed',
            'message': 'Failed to process batch request',
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/v1/model/info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model"""
    global fraud_detector
    
    if fraud_detector is None:
        return jsonify({
            'error': 'Model not loaded',
            'status': 'not_available',
            'timestamp': datetime.now().isoformat()
        }), 404
    
    try:
        model_info = fraud_detector.get_model_info()
        model_info['api_version'] = '1.0'
        model_info['timestamp'] = datetime.now().isoformat()
        
        return jsonify(model_info), 200
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve model information',
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/v1/model/features', methods=['GET'])
def get_feature_importance():
    """Get feature importance from the model"""
    global fraud_detector
    
    if fraud_detector is None or not fraud_detector.is_trained:
        return jsonify({
            'error': 'Model not available',
            'status': 'not_available',
            'timestamp': datetime.now().isoformat()
        }), 404
    
    try:
        feature_importance = fraud_detector.get_feature_importance()
        
        return jsonify({
            'feature_importance': feature_importance,
            'feature_count': len(feature_importance),
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get feature importance: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve feature importance',
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/v1/stats', methods=['GET'])
def get_api_stats():
    """Get API usage statistics"""
    return jsonify({
        'api_statistics': api_stats,
        'timestamp': datetime.now().isoformat(),
        'status': 'success'
    }), 200


@app.route('/api/v1/explain', methods=['POST'])
@timing_decorator
def explain_prediction():
    """
    Get explanation for a fraud prediction
    
    Request body:
    {
        "Amount": 150.00,
        "V1": -1.359807,
        "V2": -0.072781,
        ... (other features)
    }
    
    Response:
    {
        "fraud_probability": 0.234,
        "explanation": {
            "top_features": [...],
            "risk_factors": [...],
            "model_contributions": {...}
        },
        "status": "success"
    }
    """
    global fraud_detector
    
    try:
        # Check if model is available
        if fraud_detector is None or not fraud_detector.is_trained:
            return jsonify({
                'error': 'Fraud detection model not available',
                'status': 'service_unavailable',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        # Get and validate request data
        if not request.is_json:
            return jsonify({
                'error': 'Request must be JSON',
                'status': 'bad_request',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        transaction_data = request.get_json()
        validated_data = validate_transaction_data(transaction_data)
        
        # Get prediction with explanation
        result = fraud_detector.score_transaction(validated_data)
        
        # Get feature importance for explanation
        feature_importance = fraud_detector.get_feature_importance()
        
        # Create explanation
        explanation = {
            'top_features': list(feature_importance.items())[:10],
            'risk_factors': result.get('risk_factors', []),
            'model_contributions': result.get('model_scores', {}),
            'confidence': result.get('confidence', 0),
            'uncertainty': result.get('uncertainty', 0)
        }
        
        return jsonify({
            'fraud_probability': result['fraud_probability'],
            'risk_level': result['risk_level'],
            'recommendation': result['recommendation'],
            'explanation': explanation,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }), 200
        
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        return jsonify({
            'error': 'Failed to generate explanation',
            'message': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500


@app.before_request
def before_request():
    """Log incoming requests and set start time"""
    request.start_time = time.time()
    logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")


@app.after_request
def after_request(response):
    """Log request completion"""
    if hasattr(request, 'start_time'):
        duration = round((time.time() - request.start_time) * 1000, 2)
        logger.info(f"Response: {response.status_code} ({duration}ms)")
    return response


def initialize_fraud_detector(model_path: str = None):
    """Initialize the fraud detector"""
    global fraud_detector
    
    try:
        fraud_detector = AdvancedFraudDetector()
        
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading pre-trained model from {model_path}")
            fraud_detector.load_model(model_path)
        else:
            logger.warning("No pre-trained model found. API will run in fallback mode.")
            logger.info("Train a model and save it to enable full functionality.")
        
        logger.info("Fraud detector initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize fraud detector: {str(e)}")
        fraud_detector = None


def create_app(config: Dict[str, Any] = None):
    """Application factory function"""
    
    # Load configuration
    if config:
        app.config.update(config)
    
    # Initialize fraud detector
    model_path = os.environ.get('FRAUD_MODEL_PATH', 'models/fraud_detector.pkl')
    initialize_fraud_detector(model_path)
    
    return app


def main():
    """Main entry point for the API"""
    
    # Configuration from environment variables
    config = {
        'DEBUG': os.environ.get('DEBUG', 'False').lower() == 'true',
        'HOST': os.environ.get('HOST', '0.0.0.0'),
        'PORT': int(os.environ.get('PORT', 5000))
    }
    
    logger.info("Starting Advanced Fraud Detection API...")
    logger.info(f"Configuration: {config}")
    
    # Create and configure app
    app = create_app(config)
    
    # Start the server
    try:
        app.run(
            host=config['HOST'],
            port=config['PORT'],
            debug=config['DEBUG'],
            threaded=True
        )
    except KeyboardInterrupt:
        logger.info("API server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start API server: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()

        if not request.is_json:
            return jsonify({
                'error': 'Request must be JSON',
                'status': 'bad_request',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        transaction_data = request.get_json()
        
        # Validate transaction data
        validated_data = validate_transaction_data(transaction_data)
        
        # Score the transaction
        result = fraud_detector.score_transaction(validated_data)
        
        # Add API metadata
        result['api_version'] = '1.0'
        result['processing_time_ms'] = round((time.time() - request.start_time) * 1000, 2) if hasattr(request, 'start_time') else None
        
        return jsonify(result), 200
        
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return jsonify({
            'error': 'Validation error',
            'message': str(e),
            'status': 'bad_request',
            'timestamp': datetime.now().isoformat()
        }), 400
        
    except Exception as e:
        logger.error(f"Transaction scoring error: {str(e)}")
        return jsonify({
            'error': 'Scoring failed',
            'message': 'Failed to score transaction',
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/v1/batch_score', methods=['POST'])
@timing_decorator
def batch_score_transactions():
    """
    Score multiple transactions in batch
    
    Request body:
    {
        "transactions": [
            {"Amount": 150.00, "V1": -1.359807, ...},
            {"Amount": 75.50, "V1": 0.234567, ...},
            ...
        ]
    }
    
    Response:
    {
        "batch_results": [...],
        "summary": {
            "total_transactions": 10,
            "avg_fraud_probability": 0.234,
            "high_risk_count": 2
        },
        "status": "success"
    }
    """
    global fraud_detector
    
    try:
        # Check if model is available
        if fraud_detector is None or not fraud_detector.is_trained:
            return jsonify({
                'error': 'Fraud detection model not available',
                'status': 'service_unavailable',
                'timestamp': datetime.now().isoformat()
            }), 503
