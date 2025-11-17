# middleware.py - Updated to work with AI flow
import json
from django.http import JsonResponse
from django.utils.deprecation import MiddlewareMixin

class AuthenticationMiddleware(MiddlewareMixin):
    def process_request(self, request):
        """
        Global middleware for basic authentication control.
        - Allows all /api/ endpoints (since AI or client-side logic manages access).
        - Allows static assets and homepage.
        - Requires authentication for all other routes.
        """
        # Allow all API routes and static files
        if (request.path.startswith('/api/') or
            request.path.startswith('/static/') or
            request.path == '/'):
            return None

        # Explicitly allow the AI-driven insurance chat endpoint
        if request.path.startswith('/api/insurance-chat/'):
            return None

        # Require login for all other routes
        if not request.user.is_authenticated:
            return JsonResponse(
                {"error": "Authentication required", "login_required": True},
                status=401
            )

        return None
