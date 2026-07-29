import type { ProcessingJob } from "@/domain/api-contract";

type WorkspaceLoadingCopy = {
  progress: number;
  title: string;
  description: string;
};

export function getWorkspaceLoadingCopy(
  uploading: boolean,
  job: ProcessingJob | null,
): WorkspaceLoadingCopy {
  // A proxied upload request can remain pending until the API's background task
  // finishes, even though polling has already observed a ready job. Ready is the
  // authoritative state and must never be described as an upload in progress.
  if (job?.stage === "ready") {
    return {
      progress: 100,
      title: "Loading processed data",
      description:
        "Upload and analysis are complete. Loading the preview and quality findings.",
    };
  }

  if (uploading) {
    return {
      progress: 18,
      title: "Uploading experiment data",
      description:
        "Securely transferring the selected file to the temporary processing session.",
    };
  }

  return {
    progress: job?.progress ?? 4,
    title: "Processing experiment data",
    description: job?.message ?? "Waiting for the LabViz processing API.",
  };
}
