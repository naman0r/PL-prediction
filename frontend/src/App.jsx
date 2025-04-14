import { useState } from "react";
import {
  createBrowserRouter,
  RouterProvider,
  Outlet,
  Link,
} from "react-router-dom";
import "./App.css";

// Placeholder Components
const Home = () => (
  <div>
    <h2>Home</h2>
    <p>Welcome to the Premier League Predictor!</p>
  </div>
);
const Predict = () => (
  <div>
    <h2>Predict Match</h2>
    <p>Prediction form goes here.</p>
  </div>
);
const Results = () => (
  <div>
    <h2>Prediction Results</h2>
    <p>Results will be shown here.</p>
  </div>
);
const NotFound = () => (
  <div>
    <h2>404 Not Found</h2>
    <p>Page not found.</p>
  </div>
);

// Basic Layout with Navigation
const Layout = () => {
  return (
    <div>
      <nav>
        <ul>
          <li>
            <Link to="/">Home</Link>
          </li>
          <li>
            <Link to="/predict">Predict</Link>
          </li>
          <li>
            <Link to="/results">Results</Link>
          </li>
        </ul>
      </nav>
      <hr />
      <Outlet /> {/* Child routes will render here */}
    </div>
  );
};

const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <Home /> },
      { path: "predict", element: <Predict /> },
      { path: "results", element: <Results /> },
    ],
  },
  {
    path: "*", // Catch-all route for 404
    element: <NotFound />,
  },
]);

function App() {
  return <RouterProvider router={router} />;
}

export default App;
