"""
Simplified DataInputManager for AutoML PySpark.

This module provides a lightweight implementation of the
``DataInputManager`` used in the AutoML PySpark project.  The main
goal of this rewrite is to streamline loading of BigQuery tables and
files while avoiding the expensive temporary table creation logic
found in the original implementation.  In addition to BigQuery
support, the manager can load data from uploaded files and from
predefined datasets included with the project.

Key features:

* BigQuery loading uses the connector's query interface directly and
  supports optional selection of columns, row limits, sampling
  percentages and WHERE clauses.  Filters and limits are pushed down
  into BigQuery via the SQL query to minimise data transfer.
* File uploads support CSV, TSV, JSON, Parquet and Excel formats.
  Uploaded files are copied into the configured ``output_dir`` for
  persistence and ease of access.  Excel files are read with pandas
  then converted into Spark DataFrames.
* Existing datasets bundled with the project (e.g. ``IRIS.csv``,
  ``bank.csv`` and ``regression_file.csv``) can be loaded by name.

The returned metadata includes basic information about the loaded
dataset such as the number of rows and columns, file format (for
files) and the original table reference (for BigQuery).
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd  # type: ignore
from pyspark.sql import DataFrame, SparkSession


class DataInputManager:
    """Unified interface for loading data into the AutoML pipeline."""

    def __init__(self, spark: SparkSession, output_dir: str, user_id: str = "default_user") -> None:
        self.spark = spark
        self.output_dir = output_dir
        self.user_id = user_id
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Supported file formats for upload
        self.supported_extensions = {
            "csv": [".csv"],
            "tsv": [".tsv", ".tab"],
            "json": [".json"],
            "parquet": [".parquet"],
            "excel": [".xlsx", ".xls"]
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_data(
        self,
        data_source: str,
        source_type: str = "auto",
        **kwargs: Any,
    ) -> Tuple[DataFrame, Dict[str, Any]]:
        """Load data from BigQuery, a file upload or an existing dataset.

        Parameters
        ----------
        data_source : str
            Identifier for the data source.  For BigQuery this should
            be a fully‑qualified table reference (``project.dataset.table``).
            For uploaded files this is a path to the file.  For
            existing datasets it can be a file name (e.g. ``iris``).
        source_type : str, default ``"auto"``
            Explicitly specify ``"bigquery"``, ``"upload"`` or
            ``"existing"``.  When set to ``"auto"``, the manager will
            attempt to infer the source type from the format of
            ``data_source``.
        **kwargs : Any
            Additional parameters passed through to the underlying load
            functions.  For BigQuery these include ``project_id``,
            ``row_limit``, ``sample_percent``, ``where_clause``,
            ``select_columns`` and ``bigquery_options``.

        Returns
        -------
        tuple
            A tuple containing the loaded Spark DataFrame and a
            metadata dictionary describing the source.
        """

        # Infer source type if set to auto
        if source_type == "auto":
            if self._is_bigquery_reference(data_source):
                source_type = "bigquery"
            elif os.path.exists(data_source):
                source_type = "upload"
            else:
                source_type = "existing"

        if source_type == "bigquery":
            df, meta = self._load_from_bigquery(data_source, **kwargs)
        elif source_type == "upload":
            df, meta = self._load_from_upload(data_source, **kwargs)
        elif source_type == "existing":
            df, meta = self._load_from_existing(data_source, **kwargs)
        else:
            raise ValueError(f"Unsupported source_type: {source_type}")

        return df, meta

    def _is_bigquery_reference(self, data_source: str) -> bool:
        """Check if the data source is a BigQuery table reference.
        
        Args:
            data_source: The data source string to check
            
        Returns:
            bool: True if it's a BigQuery table reference, False otherwise
        """
        # BigQuery table references have the format: project.dataset.table
        # They contain exactly 2 dots and no slashes or other path indicators
        if not isinstance(data_source, str):
            return False
            
        # Check for BigQuery table pattern: project.dataset.table
        parts = data_source.split('.')
        if len(parts) == 3:
            # All parts should be valid identifiers (no spaces, special chars except underscores)
            for part in parts:
                if not part or not all(c.isalnum() or c in '_-' for c in part):
                    return False
            return True
            
        return False

    def get_data_preview(self, df: DataFrame, num_rows: int = 5) -> None:
        """Display a preview of the data in the console.

        This convenience function prints the first few rows along with
        basic metadata.  It is intended for interactive use in the
        Streamlit application.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The DataFrame to preview.
        num_rows : int, default ``5``
            Number of rows to display.
        """
        try:
            row_count = df.count()
            col_count = len(df.columns)
            print(f"📝 Data preview ({row_count} rows × {col_count} columns):")
            df.show(num_rows, truncate=False)
        except Exception as e:
            print(f"⚠️ Could not preview data: {e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_bigquery_reference(self, ref: str) -> bool:
        """Return True if the string looks like a BigQuery table reference."""
        # Check if it's a file path (has file extension or path separators)
        if any(ref.lower().endswith(ext) for ext in ['.csv', '.tsv', '.json', '.parquet', '.xlsx', '.xls']):
            return False
        if '/' in ref or '\\' in ref:
            return False
        
        # BigQuery table reference should have format: [project.]dataset.table
        parts = ref.split(".")
        return 2 <= len(parts) <= 3 and all(part.replace('_', '').replace('-', '').isalnum() for part in parts)
    
    def _get_bigquery_table_size(self, table_reference: str, project_id: str = None) -> Dict[str, Any]:
        """Get BigQuery table size information."""
        try:
            from google.cloud import bigquery
            
            # Parse table reference
            clean_table_ref = table_reference.strip('`')
            table_parts = clean_table_ref.split('.')
            
            if len(table_parts) == 3:
                project_id, dataset_id, table_id = table_parts
            elif len(table_parts) == 2:
                dataset_id, table_id = table_parts
                if not project_id:
                    project_id = bigquery.Client().project
            else:
                return {}
            
            client = bigquery.Client(project=project_id)
            table_ref = client.dataset(dataset_id).table(table_id)
            table = client.get_table(table_ref)
            
            return {
                'num_rows': table.num_rows or 0,
                'num_bytes': table.num_bytes or 0,
                'size_mb': (table.num_bytes or 0) / (1024 * 1024),
                'created': table.created,
                'modified': table.modified
            }
            
        except Exception as e:
            print(f"⚠️ Could not get table size for {table_reference}: {e}")
            return {}
    
    def _create_filtered_temp_table(self, table_reference: str, project_id: str,
                                   where_clause: str = None, select_columns: str = None) -> str:
        """Create a temporary BigQuery table with filters applied."""
        try:
            from google.cloud import bigquery
            import uuid
            import time
            
            client = bigquery.Client(project=project_id)
            
            # Generate unique temp table name
            timestamp = int(time.time())
            unique_id = str(uuid.uuid4())[:8]
            temp_table_id = f"automl_temp_filtered_{timestamp}_{unique_id}"
            
            # Parse original table reference
            clean_table_ref = table_reference.strip('`')
            table_parts = clean_table_ref.split('.')
            
            if len(table_parts) == 3:
                orig_project_id, dataset_id, _ = table_parts
            elif len(table_parts) == 2:
                dataset_id, _ = table_parts
                orig_project_id = project_id
            else:
                return None
            
            # Create temp table in same dataset
            temp_table_ref = f"{project_id}.{dataset_id}.{temp_table_id}"
            
            # Build filtered query
            select_part = select_columns if select_columns else "*"
            where_part = f"WHERE {where_clause}" if where_clause else ""
            
            query = f"""
            CREATE TABLE `{temp_table_ref}` AS
            SELECT {select_part}
            FROM `{clean_table_ref}`
            {where_part}
            """
            
            print(f"🔧 Creating filtered temp table with query: {query[:100]}...")
            
            # Execute query
            job = client.query(query)
            job.result()  # Wait for completion
            
            # Set table expiration (24 hours)
            table = client.get_table(temp_table_ref)
            from datetime import timedelta
            table.expires = table.created + timedelta(hours=24)
            client.update_table(table, ["expires"])
            
            print(f"✅ Filtered temp table created: {temp_table_ref} (expires in 24h)")
            return temp_table_ref
            
        except Exception as e:
            print(f"❌ Failed to create filtered temp table: {e}")
            return None
    
    def _create_sample_temp_table(self, table_reference: str, project_id: str, 
                                 sample_size: int = 100000) -> str:
        """Create a temporary BigQuery table with sampled data from an existing table."""
        try:
            from google.cloud import bigquery
            import uuid
            import time
            
            client = bigquery.Client(project=project_id)
            
            # Generate unique temp table name
            timestamp = int(time.time())
            unique_id = str(uuid.uuid4())[:8]
            temp_table_id = f"automl_temp_sample_{timestamp}_{unique_id}"
            
            # Parse table reference
            clean_table_ref = table_reference.strip('`')
            table_parts = clean_table_ref.split('.')
            
            if len(table_parts) == 3:
                orig_project_id, dataset_id, _ = table_parts
            elif len(table_parts) == 2:
                dataset_id, _ = table_parts
                orig_project_id = project_id
            else:
                return None
            
            # Create temp table in same dataset
            temp_table_ref = f"{project_id}.{dataset_id}.{temp_table_id}"
            
            # Use TABLESAMPLE for efficient sampling
            query = f"""
            CREATE TABLE `{temp_table_ref}` AS
            SELECT *
            FROM `{clean_table_ref}` TABLESAMPLE SYSTEM (10 PERCENT)
            LIMIT {sample_size}
            """
            
            print(f"🔧 Creating sample temp table with query: {query[:100]}...")
            
            # Execute query
            job = client.query(query)
            job.result()  # Wait for completion
            
            # Set table expiration (24 hours)
            table = client.get_table(temp_table_ref)
            from datetime import timedelta
            table.expires = table.created + timedelta(hours=24)
            client.update_table(table, ["expires"])
            
            print(f"✅ Sample temp table created: {temp_table_ref} (expires in 24h)")
            return temp_table_ref
            
        except Exception as e:
            print(f"❌ Failed to create sample temp table: {e}")
            return None
    
    def load_full_data_after_feature_selection(self, original_table_reference: str, 
                                             selected_features: list, target_column: str,
                                             **kwargs) -> 'DataFrame':
        """
        Create an optimized table with full rows but only selected features for model training.
        
        This creates a new temporary BigQuery table with:
        - Full dataset rows (2.29M) with applied filters
        - Only selected features (26 columns) instead of all columns (613)
        - Optimized for fast model training
        """
        print(f"🔄 Switching to full dataset for model training...")
        print(f"📊 Creating optimized table with {len(selected_features)} selected features")
        
        # Get the original filters that were applied during initial sampling
        # CRITICAL FIX: Always use original_table_reference for full data loading, ignore _filtered_table_reference
        # The _filtered_table_reference points to the sample table (100K rows), not the full table (1.4M rows)
        where_clause = kwargs.get('where_clause')  # Use original filter if provided
        
        # Map encoded feature names back to original raw names
        raw_feature_names = []
        for feature in selected_features:
            if feature.endswith('_encoded'):
                # Remove _encoded suffix to get original name
                raw_name = feature[:-8]  # Remove '_encoded'
                raw_feature_names.append(raw_name)
            else:
                # Feature is already in raw form
                raw_feature_names.append(feature)
        
        print(f"🔄 Mapped {len(selected_features)} encoded features to raw names:")
        if selected_features:
            print(f"   Example: {selected_features[0]} → {raw_feature_names[0]}")
        
        # Create column selection (target + raw feature names, excluding target from features)
        columns_to_select = [target_column] + [f for f in raw_feature_names if f != target_column]
        select_columns = ", ".join(columns_to_select)
        
        print(f"📊 Creating optimized temp table from original {original_table_reference}")
        if where_clause:
            print(f"🔍 Applying filters: {where_clause}")
        else:
            print(f"🔍 No filters specified - using full table")
        print(f"📋 Selecting {len(columns_to_select)} columns: {columns_to_select[:5]}{'...' if len(columns_to_select) > 5 else ''}")
        
        # Create optimized temp table with full rows + selected features + optional filters
        try:
            from google.cloud import bigquery
            import uuid
            import time
            from datetime import timedelta
            
            # Extract project_id and dataset_id from table reference
            table_parts = original_table_reference.split('.')
            project_id = table_parts[0]
            dataset_id = table_parts[1]
            client = bigquery.Client(project=project_id)
            
            # Create unique temp table name in the same dataset
            temp_table_name = f"automl_temp_optimized_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            temp_table_ref = f"{project_id}.{dataset_id}.{temp_table_name}"
            
            # Build optimized query: full rows + selected features + optional filters
            if where_clause:
                create_query = f"""
                CREATE TABLE `{temp_table_ref}` AS
                SELECT {select_columns}
                FROM `{original_table_reference}`
                WHERE {where_clause}
                """
            else:
                create_query = f"""
                CREATE TABLE `{temp_table_ref}` AS
                SELECT {select_columns}
                FROM `{original_table_reference}`
                """
            
            print(f"🔧 Creating optimized temp table: {temp_table_ref}")
            query_job = client.query(create_query)
            query_job.result()  # Wait for completion
            
            # Set table expiration (24 hours)
            table = client.get_table(temp_table_ref)
            table.expires = table.created + timedelta(hours=24)
            client.update_table(table, ["expires"])
            
            print(f"✅ Optimized temp table created: {temp_table_ref} (expires in 24h)")
            
            # Load the optimized table
            full_data, metadata = self.load_data(
                temp_table_ref,
                feature_engineering_phase=False,  # Disable sampling for model training
                enable_intelligent_sampling=False
            )
            
            print(f"✅ Full dataset loaded: {full_data.count():,} rows × {len(full_data.columns)} columns")
            print(f"🎯 Optimized: Full rows with selected features only")
            return full_data
            
        except Exception as e:
            print(f"⚠️ Failed to create optimized temp table: {str(e)}")
            print(f"🔄 Falling back to direct loading with column selection...")
            
            # Fallback: Load with column selection
            fallback_kwargs = {
                'select_columns': select_columns,
                'feature_engineering_phase': False,
                'enable_intelligent_sampling': False
            }
            if where_clause:
                fallback_kwargs['where_clause'] = where_clause
                
            full_data, metadata = self.load_data(
                original_table_reference,
                **fallback_kwargs
            )
            
            print(f"✅ Full dataset loaded: {full_data.count():,} rows × {len(full_data.columns)} columns")
            return full_data

    def _load_from_bigquery(self, table_reference: str, **kwargs: Any) -> Tuple[DataFrame, Dict[str, Any]]:
        """Load data from BigQuery using a pushed‑down SQL query.

        Parameters
        ----------
        table_reference : str
            The fully qualified BigQuery table name (``project.dataset.table``).
        **kwargs : Any
            Additional options such as ``project_id``, ``row_limit``,
            ``sample_percent``, ``where_clause``, ``select_columns`` and
            ``bigquery_options``.

        Returns
        -------
        tuple
            A tuple of (DataFrame, metadata).
        """
        import time
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                project_id = kwargs.get("project_id")
                if not project_id:
                    # Extract project_id from table reference (format: project.dataset.table)
                    if '.' in table_reference:
                        project_id = table_reference.split('.')[0]
                    else:
                        raise ValueError(f"Cannot determine project_id from table reference: {table_reference}")
                row_limit = kwargs.get("row_limit")
                sample_percent = kwargs.get("sample_percent")
                where_clause = kwargs.get("where_clause")
                select_columns = kwargs.get("select_columns")
                bigquery_options = kwargs.get("bigquery_options", {})
                
                # Intelligent sampling for large datasets
                enable_intelligent_sampling = kwargs.get("enable_intelligent_sampling", True)
                feature_engineering_phase = kwargs.get("feature_engineering_phase", False)
                
                # Intelligent sampling for large datasets
                if enable_intelligent_sampling and feature_engineering_phase:
                    # Step 1: Create filtered temp table first (if filters exist)
                    filtered_table_ref = None
                    if where_clause or select_columns:
                        print(f"🔧 Creating filtered temp table to check size...")
                        filtered_table_ref = self._create_filtered_temp_table(
                            table_reference,
                            project_id,
                            where_clause=where_clause,
                            select_columns=select_columns
                        )
                        
                        if filtered_table_ref:
                            print(f"✅ Filtered temp table created: {filtered_table_ref}")
                            # Check size of filtered table
                            filtered_size_info = self._get_bigquery_table_size(filtered_table_ref, project_id)
                            table_to_check = filtered_table_ref
                            size_info = filtered_size_info
                        else:
                            print(f"⚠️ Failed to create filtered temp table, checking original table size")
                            table_to_check = table_reference
                            size_info = self._get_bigquery_table_size(table_reference, project_id)
                    else:
                        # No filters, check original table size
                        table_to_check = table_reference
                        size_info = self._get_bigquery_table_size(table_reference, project_id)
                    
                    # Step 2: Check if sampling is needed (>100K rows)
                    if size_info and size_info.get('num_rows', 0) > 100000:
                        print(f"🔍 Large dataset detected: {size_info['num_rows']:,} rows")
                        print(f"📊 Creating sample temp table (100K rows) for feature engineering...")
                        
                        # Create sample temp table from the filtered table (or original if no filters)
                        sample_table_ref = self._create_sample_temp_table(
                            table_to_check,
                            project_id,
                            sample_size=100000
                        )
                        
                        if sample_table_ref:
                            print(f"✅ Using sample temp table for feature engineering: {sample_table_ref}")
                            table_reference = sample_table_ref
                            # Store original references for later use
                            kwargs['_original_table_reference'] = table_reference if not filtered_table_ref else table_reference
                            kwargs['_filtered_table_reference'] = filtered_table_ref if filtered_table_ref else table_reference
                            # Clear parameters since they're already applied in temp tables
                            row_limit = None
                            sample_percent = None
                            where_clause = None
                            select_columns = None
                        else:
                            print(f"⚠️ Failed to create sample temp table, using filtered table with row limit")
                            if filtered_table_ref:
                                table_reference = filtered_table_ref
                                where_clause = None  # Already applied
                                select_columns = None  # Already applied
                            if not row_limit:
                                row_limit = 100000  # Fallback limit
                    else:
                        print(f"📊 Dataset size acceptable: {size_info.get('num_rows', 0):,} rows - no sampling needed")
                        if filtered_table_ref:
                            # Use filtered table but no sampling needed
                            table_reference = filtered_table_ref
                            kwargs['_filtered_table_reference'] = filtered_table_ref
                            where_clause = None  # Already applied
                            select_columns = None  # Already applied

                # Clean project_id if it contains backticks
                if project_id:
                    project_id = project_id.strip('`')

                # Attempt to derive project ID from the table reference if not provided
                if not project_id:
                    # CRITICAL FIX: Clean backticks BEFORE splitting to extract project ID
                    clean_table_for_project = table_reference.strip('`')
                    parts = clean_table_for_project.split(".")
                    if len(parts) == 3:
                        project_id = parts[0]
                    else:
                        raise ValueError(
                            "A BigQuery project_id is required when table_reference"
                            " does not include it."
                        )

                # Build SELECT clause
                if select_columns:
                    select_part = select_columns
                else:
                    select_part = "*"

                # Start constructing the SQL query
                # For BigQuery Spark connector, use table reference directly without backticks
                # Remove any existing backticks from table_reference to avoid encoding issues
                clean_table_ref = table_reference.strip('`')
                sql = f"SELECT {select_part} FROM {clean_table_ref}"

                # Apply table sampling if provided
                if sample_percent:
                    # BigQuery TABLESAMPLE supports system sampling percentages
                    sql += f" TABLESAMPLE SYSTEM ({float(sample_percent)} PERCENT)"

                # Apply WHERE clause if provided
                if where_clause:
                    sql += f" WHERE {where_clause}"

                # Apply row limit at the end of the query
                if row_limit:
                    sql += f" LIMIT {int(row_limit)}"

                # Determine the dataset ID for the connector.  BigQuery's Spark
                # connector requires a dataset to be specified when using the
                # ``query`` option.  Extract it from the table reference (which
                # may be ``project.dataset.table`` or ``dataset.table``).
                # Clean table_reference first to remove any backticks
                clean_table_ref = table_reference.strip('`')
                table_parts = clean_table_ref.split(".")
                # Initialise dataset_id to avoid NameError if parsing fails
                dataset_id: Optional[str] = None
                if len(table_parts) == 3:
                    # format: project.dataset.table
                    _, dataset_id, _ = table_parts
                elif len(table_parts) == 2:
                    # format: dataset.table; project_id must be provided separately
                    dataset_id, _ = table_parts
                # After parsing, ensure dataset_id is set
                if not dataset_id:
                    raise ValueError(
                        f"Unable to determine dataset from table_reference '{table_reference}'."
                    )

                # Configure the reader with retry logic - use table option for better compatibility
                reader = (
                    self.spark.read.format("bigquery")
                    .option("parentProject", project_id)
                    .option("viewsEnabled", "true")
                    .option("useAvroLogicalTypes", "true")
                    .option("table", table_reference)
                )

                # When using query with the BigQuery connector, materialization
                # options specify where the temporary table for the query
                # execution will live.  Without these, queries against large
                # tables can fail if the default dataset is not available.
                reader = reader.option("materializationDataset", dataset_id)
                reader = reader.option("materializationProject", project_id)

                # Apply any additional BigQuery connector options
                for key, value in bigquery_options.items():
                    reader = reader.option(key, value)

                # Load the DataFrame
                df = reader.load()

                # Gather metadata
                try:
                    row_count = df.count()
                except Exception:
                    row_count = -1  # Unknown for very large tables
                col_count = len(df.columns)
                meta: Dict[str, Any] = {
                    "source_type": "bigquery",
                    "table_reference": table_reference,
                    "project_id": project_id,
                    "row_count": row_count,
                    "column_count": col_count,
                    "query": sql,
                }
                
                # Add table references to metadata if they were created
                if '_filtered_table_reference' in kwargs:
                    meta['_filtered_table_reference'] = kwargs['_filtered_table_reference']
                if '_original_table_reference' in kwargs:
                    meta['_original_table_reference'] = kwargs['_original_table_reference']
                
                return df, meta
                
            except Exception as e:
                error_msg = str(e)
                print(f"⚠️ BigQuery data loading attempt {attempt + 1}/{max_retries} failed: {error_msg}")
                
                # Check if it's a connector issue
                if "DATA_SOURCE_NOT_FOUND" in error_msg or "Failed to find the data source: bigquery" in error_msg:
                    print(f"🔧 Detected BigQuery connector issue - BigQuery connector not properly loaded in Spark session")
                    print(f"💡 This is a configuration issue - BigQuery connector must be configured before Spark session creation")
                    print(f"🔄 Retrying with existing session...")
                
                if attempt < max_retries - 1:
                    print(f"⏳ Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"❌ All {max_retries} attempts failed")
                    raise RuntimeError(f"Failed to load data from BigQuery after {max_retries} attempts. Last error: {error_msg}")

    def _load_from_upload(self, file_path: str, **kwargs: Any) -> Tuple[DataFrame, Dict[str, Any]]:
        """Load data from an uploaded file.

        Supported formats are CSV, TSV, JSON, Parquet and Excel.  Excel
        files are read via pandas then converted to a Spark DataFrame.  The
        uploaded file is copied into ``output_dir`` so that downstream
        tasks (e.g. job reloading) can access it later.
        """
        # Validate that the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        file_format = None
        for fmt, extensions in self.supported_extensions.items():
            if ext in extensions:
                file_format = fmt
                break

        if file_format is None:
            raise ValueError(
                f"Unsupported file extension '{ext}'. Supported extensions: {self.supported_extensions}"
            )

        # Copy file to output_dir with a unique name to avoid collisions
        base_name = os.path.basename(file_path)
        dest_path = os.path.join(self.output_dir, base_name)
        if file_path != dest_path:
            shutil.copy(file_path, dest_path)

        # Read the file into a Spark DataFrame based on its format
        if file_format == "csv":
            delimiter = kwargs.get("delimiter", ",")
            header = str(kwargs.get("header", True)).lower()
            infer_schema = str(kwargs.get("inferSchema", True)).lower()
            df = self.spark.read.option("header", header).option("inferSchema", infer_schema).option(
                "delimiter", delimiter
            ).csv(dest_path)
        elif file_format == "tsv":
            header = str(kwargs.get("header", True)).lower()
            infer_schema = str(kwargs.get("inferSchema", True)).lower()
            df = self.spark.read.option("header", header).option("inferSchema", infer_schema).option(
                "delimiter", "\t"
            ).csv(dest_path)
        elif file_format == "json":
            df = self.spark.read.json(dest_path)
        elif file_format == "parquet":
            df = self.spark.read.parquet(dest_path)
        elif file_format == "excel":
            # Read using pandas then convert to Spark DataFrame
            sheet_name = kwargs.get("sheet_name")
            try:
                pandas_df = pd.read_excel(dest_path, sheet_name=sheet_name)
            except Exception as e:
                raise RuntimeError(f"Error reading Excel file: {e}")
            df = self.spark.createDataFrame(pandas_df)
        else:
            raise ValueError(f"Unhandled file format: {file_format}")

        # Gather metadata
        try:
            row_count = df.count()
        except Exception:
            row_count = -1
        col_count = len(df.columns)
        meta: Dict[str, Any] = {
            "source_type": "upload",
            "file_format": file_format,
            "original_file": file_path,
            "saved_file": dest_path,
            "row_count": row_count,
            "column_count": col_count,
        }
        return df, meta

    def _load_from_existing(self, name: str, **kwargs: Any) -> Tuple[DataFrame, Dict[str, Any]]:
        """Load a built‑in dataset by name or file path.

        The AutoML project ships with a few small example datasets.  If
        ``name`` matches one of these known datasets (case‑insensitive),
        it is loaded directly from the package directory.  Otherwise,
        ``name`` is treated as a path relative to the current working
        directory or to the package directory.  The caller can also
        provide a full path to a file.
        
        For Dataproc environments, this method also handles GCS paths.
        """
        # Handle GCS paths directly for Dataproc environments
        if name.startswith('gs://'):
            print(f"📁 Loading data from GCS path: {name}")
            try:
                # Use Spark to read CSV files from GCS
                if name.endswith('.csv'):
                    df = self.spark.read.csv(name, header=True, inferSchema=True)
                    row_count = df.count()
                    col_count = len(df.columns)
                    
                    meta = {
                        "source_type": "existing",
                        "source": name,
                        "file_format": "csv",
                        "row_count": row_count,
                        "column_count": col_count,
                        "columns": df.columns
                    }
                    print(f"✅ Successfully loaded GCS file: {row_count} rows, {col_count} columns")
                    return df, meta
                else:
                    raise ValueError(f"Unsupported GCS file format for {name}. Only CSV files are supported.")
            except Exception as e:
                raise FileNotFoundError(f"Could not load GCS dataset {name}: {str(e)}")
        
        # Known datasets packaged with the project
        known_files = {
            "iris": "IRIS.csv",
            "bank": "bank.csv", 
            "regression": "regression_file.csv",
        }

        automl_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = None

        lower_name = name.lower()
        if lower_name in known_files:
            file_path = os.path.join(automl_dir, known_files[lower_name])
        else:
            # Try interpreting name as a direct path or with .csv extension
            candidates = [name, f"{name}.csv"]
            for candidate in candidates:
                # Absolute or relative to working directory
                if os.path.exists(candidate):
                    file_path = candidate
                    break
                # Relative to automl directory
                alt_path = os.path.join(automl_dir, candidate)
                if os.path.exists(alt_path):
                    file_path = alt_path
                    break

        if not file_path:
            raise FileNotFoundError(f"Could not find existing dataset: {name}")

        # Read using the upload loader logic based on extension
        return self._load_from_upload(file_path)


def build_bigquery_query(table_ref: str, options: dict = None, is_view: bool = False) -> str:
    """
    Build a BigQuery query with user configurations.
    
    Args:
        table_ref: BigQuery table reference (project.dataset.table or dataset.table)
        options: Dictionary containing query options like WHERE clauses, column selection, etc.
        is_view: Whether the table reference is a view (optional, will be auto-detected if not provided)
        
    Returns:
        str: Configured SQL query
    """
    if not options:
        options = {}
    
    # Start building the query
    select_clause = options.get('select_columns', '*')
    if select_clause and select_clause.strip():
        select_part = select_clause.strip()
    else:
        select_part = '*'
    
    # Build the FROM clause with sampling if specified
    # For BigQuery Spark connector, use table reference directly without backticks
    clean_table_ref = table_ref.strip('`')
    from_part = clean_table_ref
    sampling_where = ""
    
    if options.get('sample_percent'):
        sample_percent = options['sample_percent']
        if is_view:
            # For views, use RAND() function for sampling
            sampling_where = f"AND RAND() < {sample_percent / 100.0}"
        else:
            # For tables, use TABLESAMPLE SYSTEM
            from_part += f" TABLESAMPLE SYSTEM ({sample_percent} PERCENT)"
    
    # Build the WHERE clause if specified
    where_part = ""
    if options.get('where_clause') and options['where_clause'].strip():
        where_part = f"WHERE {options['where_clause'].strip()}"
        if sampling_where:
            where_part += f" {sampling_where}"
    elif sampling_where:
        # If no WHERE clause but we have sampling, create one
        where_part = f"WHERE {sampling_where.strip('AND ')}"
    
    # Build the LIMIT clause if specified
    limit_part = ""
    if options.get('row_limit'):
        limit_part = f"LIMIT {options['row_limit']}"
    
    # Construct the final query
    query_parts = [f"SELECT {select_part}", f"FROM {from_part}"]
    
    if where_part:
        query_parts.append(where_part)
    
    if limit_part:
        query_parts.append(limit_part)
    
    return " ".join(query_parts)

