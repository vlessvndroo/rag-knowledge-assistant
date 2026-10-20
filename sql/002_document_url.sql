-- Public URL of each source document, used to cite answers.
alter table documents add column if not exists url text;
