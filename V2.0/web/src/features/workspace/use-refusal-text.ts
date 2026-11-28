import { useTranslations } from "next-intl";
import { useCallback } from "react";

import { refusals } from "@/domain/capabilities";

type Refusal = { code: string; params: Record<string, string | number>; message: string };

/** Words an error for the user: a refusal or a service error by its code, anything else by its own message. */
export function useRefusalText() {
  const refused = useTranslations("refusals");
  const service = useTranslations("serviceErrors");
  return useCallback((cause: unknown): string => {
    const { code = "", params = {}, message = String(cause) } = (cause ?? {}) as Partial<Refusal>;
    // A figure refusal that 2.4 stored has no params: its message still says what is wrong.
    if (refusals[code]?.every((name) => name in params)) return refused(code, params);
    // A DOMException's code is a number.
    return typeof code === "string" && code && service.has(code) ? service(code) : message;
  }, [refused, service]);
}
