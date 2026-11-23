import { useTranslations } from "next-intl";

import { refusals } from "@/domain/capabilities";

type Refusal = { code: string; params: Record<string, string | number>; message: string };

/** Words an error for the user: a refusal by its code, anything else by its own message. */
export function useRefusalText() {
  const t = useTranslations("refusals");
  return (cause: unknown) => {
    const { code = "", params = {}, message = String(cause) } = (cause ?? {}) as Partial<Refusal>;
    // A figure refusal that 2.4 stored has no params: its message still says what is wrong.
    const worded = refusals[code]?.every((name) => name in params);
    return worded ? t(code, params) : message;
  };
}
