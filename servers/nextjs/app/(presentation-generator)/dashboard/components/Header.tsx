"use client";

import Wrapper from "@/components/Wrapper";
import React from "react";
import Link from "next/link";
import BackBtn from "@/components/BackBtn";
import { usePathname, useRouter } from "next/navigation";
import HeaderNav from "@/app/(presentation-generator)/components/HeaderNab";
import { Layout, FilePlus2 } from "lucide-react";
import { trackEvent, MixpanelEvent } from "@/utils/mixpanel";
import { useTranslations, useLocale } from 'next-intl';
const Header = () => {
  const pathname = usePathname();
  const router = useRouter();
  const locale = useLocale();

  const handleLocaleChange = (newLocale: string) => {
    // pathname might be /en/some-page or /zh/some-page or just /
    // We need to remove the existing locale prefix if it exists
    const newPath = pathname.startsWith(`/${locale}`)
      ? pathname.substring(`/${locale}`.length) || '/'
      : pathname;

    router.replace(`/${newLocale}${newPath}`);
  };

  return (
    <div className="bg-[#5146E5] w-full shadow-lg sticky top-0 z-50">
      <Wrapper>
        <div className="flex items-center justify-between py-1">
          <div className="flex items-center gap-3">
            {(pathname !== "/upload" && pathname !== "/dashboard") && <BackBtn />}
            <Link href="/dashboard" onClick={() => trackEvent(MixpanelEvent.Navigation, { from: pathname, to: "/dashboard" })}>
              <img
                src="/logo-white.png"
                alt="Presentation logo"
                className="h-16"
              />
            </Link>
          </div>
          <div className="flex items-center gap-3">
            <Link
              href="/custom-template"
              prefetch={false}
              onClick={() => trackEvent(MixpanelEvent.Navigation, { from: pathname, to: "/custom-template" })}
              className="flex items-center gap-2 px-3 py-2 text-white hover:bg-primary/80 rounded-md transition-colors outline-none"
              role="menuitem"
            >
              <FilePlus2 className="w-5 h-5" />
              <span className="text-sm font-medium font-inter">Create Template</span>
            </Link>
            <Link
              href="/template-preview"
              prefetch={false}
              onClick={() => trackEvent(MixpanelEvent.Navigation, { from: pathname, to: "/template-preview" })}
              className="flex items-center gap-2 px-3 py-2 text-white hover:bg-primary/80 rounded-md transition-colors outline-none"
              role="menuitem"
            >
              <Layout className="w-5 h-5" />
              <span className="text-sm font-medium font-inter">Templates</span>
            </Link>
            
            {/* Language Switcher */}
            <div className="flex items-center gap-2 border-l border-white/20 pl-3 ml-1">
              <button
                onClick={() => handleLocaleChange('en')}
                className={`text-sm font-medium px-2 py-1 rounded transition-colors ${locale === 'en' ? 'bg-white/20 text-white' : 'text-white/70 hover:text-white hover:bg-white/10'}`}
              >
                EN
              </button>
              <span className="text-white/30">|</span>
              <button
                onClick={() => handleLocaleChange('zh')}
                className={`text-sm font-medium px-2 py-1 rounded transition-colors ${locale === 'zh' ? 'bg-white/20 text-white' : 'text-white/70 hover:text-white hover:bg-white/10'}`}
              >
                中文
              </button>
            </div>
            
            <HeaderNav />
          </div>
        </div>
      </Wrapper>
    </div>
  );
};

export default Header;
