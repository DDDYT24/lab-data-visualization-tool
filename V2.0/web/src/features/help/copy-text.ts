export async function copyText(value: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
      return true;
    }
  } catch { /* Try the local document fallback when clipboard permission is denied. */ }
  const focused = document.activeElement;
  const selection = window.getSelection();
  const ranges = selection ? Array.from({length: selection.rangeCount}, (_, i) => selection.getRangeAt(i).cloneRange()) : [];
  const inputSelection = focused instanceof HTMLInputElement || focused instanceof HTMLTextAreaElement
    ? {start: focused.selectionStart, end: focused.selectionEnd, direction: focused.selectionDirection} : null;
  const textarea = document.createElement("textarea");
  textarea.value = value;
  textarea.readOnly = true;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  try {
    document.body.appendChild(textarea);
    textarea.select();
    return document.execCommand("copy");
  } catch { return false; }
  finally {
    textarea.remove();
    if (focused instanceof HTMLElement && focused.isConnected) focused.focus({preventScroll:true});
    if (inputSelection && inputSelection.start !== null && inputSelection.end !== null) {
      (focused as HTMLInputElement | HTMLTextAreaElement).setSelectionRange(inputSelection.start, inputSelection.end, inputSelection.direction ?? undefined);
    }
    if (selection) { selection.removeAllRanges(); ranges.forEach(range => selection.addRange(range)); }
  }
}
