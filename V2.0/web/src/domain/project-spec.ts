import { z } from "zod";

import projectSpecSchemaDocument from "../../../contracts/project-spec-v1.schema.json";

export const projectSpecSchema = z.fromJSONSchema(
  projectSpecSchemaDocument as Parameters<typeof z.fromJSONSchema>[0],
);
export type ProjectSpec = z.infer<typeof projectSpecSchema>;
