// Temporarily disabled next-intl middleware
// import createMiddleware from 'next-intl/middleware';
//
// export default createMiddleware({
//   // A list of all locales that are supported
//   locales: ['en', 'zh'],
//
//   // Used when no locale matches
//   defaultLocale: 'en'
// });

import { NextRequest, NextResponse } from 'next/server';

export default function middleware(request: NextRequest) {
  const pathname = request.nextUrl.pathname;

  // Define presentation-generator routes that should not be redirected
  const presentationRoutes = [
    '/upload', '/dashboard', '/outline', '/presentation', '/settings',
    '/custom-template', '/documents-preview', '/pdf-maker', '/template-preview'
  ];

  // Check if current path is a presentation-generator route
  const isPresentationRoute = presentationRoutes.some(route =>
    pathname === route || pathname.startsWith(route + '/')
  );

  // Redirect root to /en
  if (pathname === '/') {
    return NextResponse.redirect(new URL('/en', request.url));
  }

  // If path doesn't start with /en or /zh, and is not a presentation route or api route, redirect to /en + path
  if (!pathname.startsWith('/en') && !pathname.startsWith('/zh') &&
      !pathname.startsWith('/api') && !pathname.startsWith('/_next') &&
      !isPresentationRoute) {
    return NextResponse.redirect(new URL('/en' + pathname, request.url));
  }

  return NextResponse.next();
}

export const config = {
  // Match only internationalized pathnames
  matcher: ['/', '/((?!api|_next|_vercel|.*\\..*).*)']
};
