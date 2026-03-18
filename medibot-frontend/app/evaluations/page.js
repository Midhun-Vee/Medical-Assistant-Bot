"use client";

import React, { useEffect, useState } from "react";

import {
Table,
TableBody,
TableCell,
TableHead,
TableHeader,
TableRow,
} from "@/components/ui/table";

import { Badge } from "@/components/ui/badge";
import Navbar from "@/components/navbar/Navbar";

const API_URL =
process.env.NEXT_PUBLIC_API_URL ||
"http://localhost:8000";

const Evaluation = () => {
const [evaluations, setEvaluations] = useState([]);
const [loading, setLoading] = useState(true);
const [error, setError] = useState("");

useEffect(() => {
const fetchEvaluations = async () => {
try {
setLoading(true);

    const response = await fetch(
      `${API_URL}/evaluations`
    );

    if (!response.ok) {
      throw new Error(
        "Failed to fetch evaluations"
      );
    }

    const data = await response.json();

    setEvaluations(
      data.evaluations || []
    );
  } catch (error) {
    console.error(error);

    setError(
      "Unable to load evaluations."
    );
  } finally {
    setLoading(false);
  }
};

fetchEvaluations();

}, []);

const formatScore = (value) => {
if (
value === null ||
value === undefined
) {
return "—";
}

return `${Math.round(value * 100)}%`;

};

const getScoreColor = (value) => {
if (
value === null ||
value === undefined
) {
return "text-muted-foreground";
}

if (value >= 0.8) {
  return "text-green-600";
}

if (value >= 0.6) {
  return "text-yellow-600";
}

return "text-red-600";

};

if (loading) {
return ( <div className="p-6"> <h1 className="text-2xl font-semibold">
Evaluation </h1>

    <p className="mt-4 text-muted-foreground">
      Loading evaluations...
    </p>
  </div>
);

}

if (error) {
return ( <div className="p-6"> <h1 className="text-2xl font-semibold">
Evaluation </h1>

    <div className="mt-6 rounded-lg border border-red-200 bg-red-50 p-4 text-red-700">
      {error}
    </div>
  </div>
);

}

return ( 

<>
  <Navbar />
  
  <div className="space-y-6 p-6">
  {/* Header */}

  <div>
    <h1 className="text-2xl font-semibold tracking-tight">
      RAG Evaluation
    </h1>

    <p className="mt-1 text-sm text-muted-foreground">
      Evaluate retrieval quality and generated
      answer quality.
    </p>
  </div>


  {/* Summary */}

  <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">

    <div className="rounded-lg border bg-card p-4">
      <p className="text-sm text-muted-foreground">
        Total Evaluations
      </p>

      <p className="mt-2 text-2xl font-semibold">
        {evaluations.length}
      </p>
    </div>


    <div className="rounded-lg border bg-card p-4">
      <p className="text-sm text-muted-foreground">
        Evaluated
      </p>

      <p className="mt-2 text-2xl font-semibold">
        {
          evaluations.filter(
            (item) =>
              item.evaluation?.status ===
              "evaluated"
          ).length
        }
      </p>
    </div>


    <div className="rounded-lg border bg-card p-4">
      <p className="text-sm text-muted-foreground">
        No Retrieval Data
      </p>

      <p className="mt-2 text-2xl font-semibold">
        {
          evaluations.filter(
            (item) =>
              item.evaluation?.status ===
              "no_data"
          ).length
        }
      </p>
    </div>

  </div>


  {/* Evaluation Table */}

  <div className="rounded-lg border bg-card">

    <div className="border-b px-6 py-4">
      <h2 className="font-semibold">
        Evaluation Results
      </h2>

      <p className="text-sm text-muted-foreground">
        RAG evaluation metrics for each query.
      </p>
    </div>


    {evaluations.length === 0 ? (

      <div className="p-10 text-center text-sm text-muted-foreground">
        No evaluations available yet.
      </div>

    ) : (

      <div className="overflow-x-auto">

        <Table>

          <TableHeader>

            <TableRow>

              <TableHead className="min-w-[300px]">
                Question
              </TableHead>

              <TableHead>
                Status
              </TableHead>

              <TableHead className="text-center">
                Chunks
              </TableHead>

              <TableHead className="text-center">
                Faithfulness
              </TableHead>

              <TableHead className="text-center">
                Context Recall
              </TableHead>

              <TableHead className="text-center">
                Context Precision
              </TableHead>

              <TableHead className="text-center">
                Answer Correctness
              </TableHead>

              <TableHead className="text-center">
                Answer Relevancy
              </TableHead>

            </TableRow>

          </TableHeader>


          <TableBody>

            {evaluations
              .slice()
              .reverse()
              .map((item, index) => {

                const evaluation =
                  item.evaluation;

                const isEvaluated =
                  evaluation?.status ===
                  "evaluated";

                return (

                  <TableRow
                    key={`${item.chat_id}-${index}`}
                  >

                    {/* Question */}

                    <TableCell>

                      <div className="max-w-[400px]">

                        <p className="font-medium">
                          {item.question}
                        </p>

                        <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">
                          {item.answer ||
                            "No answer generated."}
                        </p>

                      </div>

                    </TableCell>


                    {/* Status */}

                    <TableCell>

                      {isEvaluated ? (

                        <Badge
                          variant="default"
                          className="bg-green-600 hover:bg-green-600"
                        >
                          Evaluated
                        </Badge>

                      ) : (

                        <Badge
                          variant="secondary"
                        >
                          No Data
                        </Badge>

                      )}

                    </TableCell>


                    {/* Retrieved chunks */}

                    <TableCell className="text-center">

                      {item.retrieved_chunks ??
                        0}

                    </TableCell>


                    {/* Faithfulness */}

                    <TableCell
                      className={`text-center font-semibold ${getScoreColor(
                        evaluation?.faithfulness
                      )}`}
                    >
                      {formatScore(
                        evaluation?.faithfulness
                      )}
                    </TableCell>


                    {/* Context Recall */}

                    <TableCell
                      className={`text-center font-semibold ${getScoreColor(
                        evaluation?.context_recall
                      )}`}
                    >
                      {formatScore(
                        evaluation?.context_recall
                      )}
                    </TableCell>


                    {/* Context Precision */}

                    <TableCell
                      className={`text-center font-semibold ${getScoreColor(
                        evaluation?.context_precision
                      )}`}
                    >
                      {formatScore(
                        evaluation?.context_precision
                      )}
                    </TableCell>


                    {/* Answer Correctness */}

                    <TableCell
                      className={`text-center font-semibold ${getScoreColor(
                        evaluation?.answer_correctness
                      )}`}
                    >
                      {formatScore(
                        evaluation?.answer_correctness
                      )}
                    </TableCell>


                    {/* Answer Relevancy */}

                    <TableCell
                      className={`text-center font-semibold ${getScoreColor(
                        evaluation?.answer_relevancy
                      )}`}
                    >
                      {formatScore(
                        evaluation?.answer_relevancy
                      )}
                    </TableCell>

                  </TableRow>

                );
              })}

          </TableBody>

        </Table>

      </div>

    )}

  </div>

</div>
</>
);
};

export default Evaluation;
