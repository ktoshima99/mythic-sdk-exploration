from pathlib import Path
from typing import Any

import dash
import plotly.graph_objects as go
from dash import MATCH, Input, Output, State, callback, ctx, dash_table, dcc, html
from numpy.typing import NDArray

from vnnort.quantizer.report.report import QuantizationReport
from vnnort.quantizer.report.viewer.report_figures import (
    generate_channel_histogram,
    generate_distribution_figure,
    layerwise_metrics_figure,
)

LAYER_OVERVIEW_TABLE_ID = "layer-overview-table"
LAYER_VISUALIZATION_DIV_ID = "layer-visualization-div"


class ReportViewer:
    """The ReportViewer class may be used to start an interactive webserver, to visualize a quantization report."""

    def __init__(self, path: str | Path, port: int = 8050):
        """Initialize the webserver on the report file at path on port port."""
        self.app = dash.Dash("Quantization Report")
        self.port = port
        self.report = QuantizationReport.load(path)
        self.load_main_layout()
        self.register_callbacks()

    def run(self) -> None:
        """Start the webserver."""
        self.app.run_server(debug=True, port=self.port)

    def load_main_layout(self) -> None:
        """Initialize the main layout."""
        self.app.title = "Quantization Report"
        if self.report.layer_metrics is not None:
            layer_wise_metrics_graph = dcc.Graph(
                figure=layerwise_metrics_figure(
                    self.report.layer_metrics["fp32_result"],
                    self.report.layer_metrics["layer_wise_quantization"],
                )
            )

        else:
            layer_wise_metrics_graph = None

        self.app.layout = html.Div(
            [
                html.Div(
                    [
                        html.H1("Layer Overview"),
                        layer_wise_metrics_graph,
                        self.layer_overview_table(),
                    ],
                    style={"width": "30%", "display": "inline-block", "verticalAlign": "top"},
                ),
                html.Div(
                    [html.Div(id=LAYER_VISUALIZATION_DIV_ID)],
                    style={"width": "70%", "display": "inline-block", "verticalAlign": "top"},
                ),
            ]
        )

    def register_callbacks(self) -> None:
        """Register all required callbacks to the dash application."""

        @callback(
            Output(LAYER_VISUALIZATION_DIV_ID, "children"),
            Input(LAYER_OVERVIEW_TABLE_ID, "active_cell"),
            Input(LAYER_OVERVIEW_TABLE_ID, "derived_viewport_data"),
        )  # type: ignore
        def update_graphs(active_cell: Any, derived_viewport_data: Any) -> html.Div:
            """Update the layer information page everytime a row in the layer table is marked."""
            if active_cell and derived_viewport_data:
                row_idx = active_cell["row"]  # Get the row index from active_cell
                row = derived_viewport_data[row_idx]  # Get the row data
                node_name = row["name"]
            else:
                # Just choose the first entry
                node_name = next(iter(self.report.network_graph.nodes.keys()))
            return self.layer_visualization_page(node_name)

        @callback(
            Output({"type": "histogram-figure", "tensor_name": MATCH}, "figure"),
            Input({"type": "channel-input", "tensor_name": MATCH}, "value"),
            State({"type": "channel-input", "tensor_name": MATCH}, "tensor_name"),
        )  # type: ignore
        def histogram_update(value: int, tensor_name: str) -> go.Figure:
            """Update the histogram figure to the histogram at index, once a new channel is selected."""
            tensor_name = ctx.inputs_list[0]["id"]["tensor_name"]
            channel_index = int(value)
            quant_info = self.report.tensor_statistics[tensor_name]
            return generate_channel_histogram(quant_info, channel=channel_index)  # type: ignore

    def layer_overview_table(self) -> html.Div:
        """Generate the layer overview table at the lefthand side of the window.

        Returns:
            html.Div: A Div containing the layer overview table
        """
        network_graph = self.report.network_graph
        nodes = network_graph.nodes
        data = [
            {
                "idx": index,
                "op_type": node.op_type,
                "name": node.name,
            }
            for index, node in enumerate(nodes.values())
        ]
        columns = [
            {"name": "Layer", "id": "idx"},
            {"name": "Type", "id": "op_type"},
            {"name": "Name", "id": "name"},
        ]

        if self.report.layer_metrics is not None:
            metrics = list(self.report.layer_metrics["layer_wise_quantization"].values())
            for index, metric in enumerate(metrics):
                data[index]["metrics"] = float(metric)
            columns.append({"name": "Metric", "id": "metrics", "type": "numeric"})

        return html.Div(
            [
                dash_table.DataTable(  # type:ignore
                    id=LAYER_OVERVIEW_TABLE_ID,
                    columns=columns,
                    data=data,
                    editable=True,
                    filter_action="native",
                    sort_action="native",
                    selected_columns=[],
                    selected_rows=[],
                    page_action="native",
                    page_current=0,
                    page_size=30,
                    style_cell={
                        "textAlign": "left",
                        "overflow": "hidden",
                        "textOverflow": "ellipsis",
                        "whiteSpace": "nowrap",
                        "maxWidth": "150px",  # Set the maximum width for columns
                    },
                )
            ],
        )

    def layer_visualization_page(self, node_name: str) -> html.Div:
        """Generate the visualization page of the layer node_name, consisting of its inputs and outputs.

        Args:
            node_name (str): Name of the layer, which will be visualized

        Returns:
            html.Div: Div containing the layer visualizations
        """
        # Extract quantization data for inputs and outputs
        node = self.report.network_graph.nodes[node_name]
        input_tensor_names = [t.name for t in node.inputs]
        output_tensor_names = [t.name for t in node.outputs]
        tensor_statistics = self.report.tensor_statistics
        input_statistics = [tensor_statistics[name] for name in input_tensor_names if name in tensor_statistics]
        output_statistics = [tensor_statistics[name] for name in output_tensor_names if name in tensor_statistics]

        layout = html.Div(
            [
                html.H1(f"Layer: {node_name}"),
                # All Inputs
                html.Details(
                    [
                        html.Summary(html.Span("Inputs", style={"font-size": "25px"})),
                        html.Div(
                            [
                                self.tensor_graphs(name, statistics)
                                for name, statistics in zip(input_tensor_names, input_statistics)
                            ]
                        ),
                    ]
                ),
                # All Outputs
                html.Details(
                    [
                        html.Summary(html.Span("Outputs", style={"font-size": "25px"})),
                        html.Div(
                            [
                                self.tensor_graphs(name, statistics)
                                for name, statistics in zip(input_tensor_names, output_statistics)
                            ]
                        ),
                    ]
                ),
            ],
        )
        return layout

    def tensor_graphs(self, tensor_name: str, statistics: dict[str, NDArray[Any]]) -> html.Div:
        """Generate the graphs for one tensor consisting of channel distribution figure and and channelwise histogram.

        Args:
            tensor_name (str): Name of the tensor
            statistics (dict[str, NDArray[Any]]): Statistics of the tensor

        Returns:
            html.Div: Div containing the figures.
        """
        all_channels_graph_id = f"{tensor_name}_all_channels_graph"

        # Style for the surrounding divs, resulting in the boxed effect around graphs
        style = {
            "width": "90%",
            "height": "500px",  # Set a fixed height
            "display": "inline-block",
            "verticalAlign": "top",
            "border": "2px solid black",
            "padding": "10px",
            "margin": "10px",
            "borderRadius": "5px",
            "boxShadow": "2px 2px 5px grey",
        }
        # n_channels = statistics["max_values"].shape[0]

        return html.Div(
            [
                html.H3(f"Tensor: {tensor_name}"),
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Graph(
                                    id=all_channels_graph_id,
                                    figure=generate_distribution_figure(tensor_name, statistics),
                                )
                            ],
                            style=style,
                        ),
                        # TODO: Check if we still need this channel specific histogram visualizations
                        # We keep this here for now as it might be useful in the future
                        # html.Div(
                        #     [
                        #         dcc.Graph(
                        #             id={"type": "histogram-figure", "tensor_name": tensor_name},
                        #             figure=generate_channel_histogram(quant_info, channel=0),
                        #         ),
                        #         html.Div(
                        #             [
                        #                 html.Label("Channel:"),
                        #                 dcc.Input(
                        #                     id={"type": "channel-input", "tensor_name": tensor_name},
                        #                     type="number",
                        #                     value=0,
                        #                     step=1,
                        #                     min=0,
                        #                     max=n_channels,
                        #                 ),
                        #                 html.Label(id="max-channel-label", children=n_channels),
                        #             ]
                        #         ),
                        #     ],
                        #     style=style,
                        # ),
                    ]
                ),
            ],
            style={"width": "100%", "display": "inline-block", "padding": "10px", "verticalAlign": "top"},
        )
