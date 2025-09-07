"use client";

import React from "react";
import Link from "next/link";
import { Layout, Plus } from "lucide-react";
import { useTranslations, useLocale } from 'next-intl';
import { usePathname, useRouter } from "next/navigation";

const Header: React.FC = () => {
  const t = useTranslations('Header');
  const locale = useLocale();
  const router = useRouter();
  const pathname = usePathname();

  const handleLocaleChange = (newLocale: string) => {
    // pathname might be /en/some-page or /zh/some-page or just /
    // We need to remove the existing locale prefix if it exists
    const newPath = pathname.startsWith(`/${locale}`)
      ? pathname.substring(`/${locale}`.length) || '/'
      : pathname;

    router.replace(`/${newLocale}${newPath}`);
  };

  return (
    <header className="w-full border-b bg-white/60 backdrop-blur supports-[backdrop-filter]:bg-white/60 sticky top-0 z-50">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <img src="/logo-white.png" alt="Presenton" className="h-6 w-auto" />
          </Link>

          <div className="flex items-center gap-6">
            <nav className="flex items-center gap-4">
              <Link href="/custom-layout" className="inline-flex items-center gap-2 text-gray-700 hover:text-gray-900">
                <Plus className="w-5 h-5" />
                <span className="text-sm font-medium font-inter">{t('createTemplate')}</span>
              </Link>
              <Link href="/template-preview" className="inline-flex items-center gap-2 text-gray-700 hover:text-gray-900">
                <Layout className="w-5 h-5" />
                <span className="text-sm font-medium font-inter">{t('templates')}</span>
              </Link>
            </nav>

            <div className="flex items-center gap-2 border-l border-gray-200 pl-4">
              <button
                onClick={() => handleLocaleChange('en')}
                className={`text-sm font-medium ${locale === 'en' ? 'text-blue-600' : 'text-gray-500 hover:text-gray-900'}`}
              >
                EN
              </button>
              <span className="text-gray-300">|</span>
              <button
                onClick={() => handleLocaleChange('zh')}
                className={`text-sm font-medium ${locale === 'zh' ? 'text-blue-600' : 'text-gray-500 hover:text-gray-900'}`}
              >
                中文
              </button>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
