use std::{collections::HashMap, path::{Path, PathBuf}};

#[derive(Default)]
pub struct Downloads {
    active: Vec<(PathBuf, String, String)>,
    cancelled: HashMap<String, usize>,
    unresolved: HashMap<String, usize>,
}

impl Downloads {
    pub fn cancelled(&mut self, url: &str) {
        // Wry also reports a terminal event for a locally rejected request.
        *self.cancelled.entry(url.to_owned()).or_default() += 1;
    }

    pub fn start(&mut self, path: PathBuf, url: String, id: String) -> bool {
        if self.active.iter().any(|(target, _, _)| target == &path) {
            return false;
        }
        self.active.push((path, url, id));
        true
    }

    pub fn finish(&mut self, url: &str, path: Option<&Path>) -> Vec<String> {
        let mut ended = Vec::new();
        if let Some(path) = path {
            let Some(index) = self.active.iter().position(|(target, source, _)| source == url && target == path) else { return ended; };
            ended.push(self.active.remove(index).2);
        } else {
            let count = self.active.iter().filter(|(_, source, _)| source == url).count()
                + self.cancelled.get(url).copied().unwrap_or(0);
            if count == 0 { return ended; }
            *self.unresolved.entry(url.to_owned()).or_default() += 1;
        }
        // Interrupted WebView2 transfers have no destination. Keep all ambiguous
        // siblings busy until every request for this URL has a terminal event.
        let remaining = self.active.iter().filter(|(_, source, _)| source == url).count()
            + self.cancelled.get(url).copied().unwrap_or(0);
        if self.unresolved.get(url).copied().unwrap_or(0) == remaining {
            self.active.retain(|(_, source, id)| {
                if source == url { ended.push(id.clone()); false } else { true }
            });
            self.cancelled.remove(url);
            self.unresolved.remove(url);
        }
        ended
    }
}

pub fn matches_extension(path: &Path, extension: &str) -> bool {
    extension.is_empty()
        || path.extension().and_then(|s| s.to_str()).unwrap_or("").eq_ignore_ascii_case(extension)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_without_path_releases_only_its_transfer_and_is_idempotent() {
        let mut downloads = Downloads::default();
        assert!(downloads.start("一.pdf".into(), "one".into(), "1".into()));
        assert!(downloads.start("二.zip".into(), "two".into(), "2".into()));
        assert_eq!(downloads.finish("one", None), ["1"]);
        assert!(downloads.finish("one", None).is_empty());
        assert_eq!(downloads.finish("two", Some(Path::new("二.zip"))), ["2"]);
        assert!(downloads.active.is_empty());
    }

    #[test]
    fn same_source_failures_cannot_release_a_still_running_sibling() {
        let mut downloads = Downloads::default();
        assert!(downloads.start("export.pdf".into(), "one".into(), "1".into()));
        assert!(downloads.start("other.pdf".into(), "one".into(), "2".into()));
        assert!(!downloads.start("export.pdf".into(), "two".into(), "3".into()));
        assert!(downloads.finish("one", None).is_empty());
        assert_eq!(downloads.active.len(), 2);
        assert_eq!(downloads.finish("one", Some(Path::new("other.pdf"))), ["2", "1"]);
        assert!(downloads.start("export.pdf".into(), "one".into(), "4".into()));
    }

    #[test]
    fn unknown_or_mismatched_completions_leave_active_work_protected() {
        let mut downloads = Downloads::default();
        downloads.start("keep.pdf".into(), "one".into(), "1".into());
        assert!(downloads.finish("foreign", None).is_empty());
        assert!(downloads.finish("one", Some(Path::new("other.pdf"))).is_empty());
        assert_eq!(downloads.active.len(), 1);
    }

    #[test]
    fn cancelling_a_save_dialog_cannot_release_another_download_of_the_same_url() {
        for cancel_finishes_first in [true, false] {
            let mut downloads = Downloads::default();
            downloads.start("keep.pdf".into(), "one".into(), "1".into());
            downloads.cancelled("one");
            if cancel_finishes_first {
                assert!(downloads.finish("one", None).is_empty());
                assert_eq!(downloads.finish("one", Some(Path::new("keep.pdf"))), ["1"]);
            } else {
                assert_eq!(downloads.finish("one", Some(Path::new("keep.pdf"))), ["1"]);
                assert!(downloads.finish("one", None).is_empty());
            }
            assert!(downloads.active.is_empty());assert!(downloads.cancelled.is_empty());
        }
    }

    #[test]
    fn two_interrupted_siblings_are_released_only_after_both_terminal_events() {
        let mut downloads = Downloads::default();
        downloads.start("one.pdf".into(), "one".into(), "1".into());
        downloads.start("two.pdf".into(), "one".into(), "2".into());
        assert!(downloads.finish("one", None).is_empty());
        assert_eq!(downloads.finish("one", None), ["1", "2"]);
    }

    #[test]
    fn all_export_formats_keep_their_extension_on_chinese_long_paths() {
        for extension in ["png", "svg", "pdf", "csv", "json", "zip"] {
            let path = PathBuf::from(format!("C:/中文路径/{}/科研结果.{}", "路径".repeat(150), extension.to_uppercase()));
            assert!(matches_extension(&path, extension));
            assert!(!matches_extension(Path::new("wrong.txt"), extension));
            assert!(!matches_extension(Path::new("no-extension"), extension));
        }
    }
}
